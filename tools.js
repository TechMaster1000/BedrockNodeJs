"""Bedrock service for handling Converse API calls with streaming only"""

import asyncio
import json
import queue
import threading
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, AsyncGenerator, Optional
from models import Message, ChatRequest
from config import config

logger = logging.getLogger(__name__)

class TokenCache:
    """Token cache manager with TTL and cleanup"""
    
    def __init__(self):
        self.cache = {}  # {email: {"token": str, "expires_at": datetime}}
        self.lock = asyncio.Lock()
        self.cleanup_task = None
        
    async def start_cleanup_task(self):
        """Start background task to cleanup expired tokens"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Periodically remove expired tokens from cache"""
        while True:
            try:
                await asyncio.sleep(300)  # Run cleanup every 5 minutes
                async with self.lock:
                    now = datetime.now()
                    expired_emails = [
                        email for email, data in self.cache.items()
                        if data["expires_at"] < now
                    ]
                    for email in expired_emails:
                        del self.cache[email]
                        logger.info(f"Removed expired token for user: {email}")
                    
                    if expired_emails:
                        logger.info(f"Cleaned up {len(expired_emails)} expired tokens")
            except Exception as e:
                logger.error(f"Error in token cleanup task: {e}")
    
    async def get_cached_token(self, email: str) -> Optional[Dict[str, Any]]:
        """Get cached token if valid"""
        async with self.lock:
            if email in self.cache:
                token_data = self.cache[email]
                if token_data["expires_at"] > datetime.now():
                    time_remaining = (token_data["expires_at"] - datetime.now()).total_seconds()
                    logger.info(f"Using cached token for {email}, expires in {time_remaining:.0f} seconds")
                    return token_data
                else:
                    # Token expired, remove it
                    del self.cache[email]
                    logger.info(f"Cached token for {email} has expired, removing from cache")
            return None
    
    async def set_token(self, email: str, token: str, valid_for_ms: int):
        """Cache a new token with expiration"""
        async with self.lock:
            # Calculate expiration with a small buffer (30 seconds before actual expiry)
            expires_at = datetime.now() + timedelta(milliseconds=valid_for_ms - 30000)
            self.cache[email] = {
                "token": token,
                "expires_at": expires_at
            }
            logger.info(f"Cached new token for {email}, expires at {expires_at}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cached_users": len(self.cache),
            "users": list(self.cache.keys())
        }

class BedrockService:
    def __init__(self, bedrock_manager):
        self.bedrock_manager = bedrock_manager
        self.tools = self._initialize_tools()
        # System prompt for the assistant
        self.system_prompt = self._get_system_prompt()
        # Initialize token cache
        self.token_cache = TokenCache()
        # Start cleanup task when service initializes
        asyncio.create_task(self.token_cache.start_cleanup_task())
    
    def _get_system_prompt(self) -> List[Dict[str, Any]]:
        """Define the system prompt for the assistant"""
        return [
            {
                "text": """You are a helpful AI assistant with access to a knowledge base through the search_passages tool. 

When users ask questions that might benefit from specific information, documentation, or policies, use the search_passages tool to retrieve relevant information before answering.

CRITICAL RULES FOR TOOL USAGE AND ACCURACY:

1. **NEVER MAKE UP INFORMATION**: 
   - When you use the search_passages tool, base your response ONLY on the actual search results returned
   - Do NOT fabricate, imagine, or create information that isn't in the search results
   - Do NOT add details, examples, or explanations beyond what's provided in the search results
   - If the search results don't contain specific information, explicitly say so

2. **STRICT ADHERENCE TO SEARCH RESULTS**:
   - Quote or paraphrase ONLY what is actually present in the retrieved passages
   - When citing information, be precise about what the source actually says
   - Do not extrapolate or infer beyond what is explicitly stated in the passages
   - If information is partial or unclear in the results, acknowledge this limitation

3. **HANDLING INSUFFICIENT RESULTS**:
   - If search results are empty or irrelevant, clearly state: "I couldn't find relevant information in the knowledge base for this query"
   - If results are partial, say what you found and what is missing: "The knowledge base contains X but doesn't have information about Y"
   - Never fill gaps with assumed or general knowledge when using tools
   - Offer to search again with different terms if the initial search was unsuccessful

4. **RESPONSE STRUCTURE WHEN USING TOOLS**:
   - Start with what was actually found (or not found) in the search
   - Present only the information from the search results
   - If relevant passages exist, synthesize them accurately without adding external information
   - Always distinguish between what comes from search results and any general context

CRITICAL FORMATTING RULES:
You MUST format your responses using proper HTML tags for a professional appearance. NEVER use \n or \n\n for line breaks - use HTML tags instead.

1. **Structure and Spacing**:
   - Use <p> tags for paragraphs (these automatically create spacing)
   - Use <br> for single line breaks only when needed within a paragraph
   - Use <hr> for section dividers when appropriate
   - NEVER use \n or \n\n - let HTML tags handle all spacing

2. **Text Emphasis**:
   - Use <strong> or <b> for important terms, headings, and key points
   - Use <em> or <i> for emphasis on specific words
   - Use <u> for underlined text when highlighting critical information

3. **Lists and Organization**:
   - Use <ul> and <li> for unordered lists
   - Use <ol> and <li> for numbered/ordered lists
   - Use <dl>, <dt>, and <dd> for definition lists when explaining terms

4. **Headings and Sections**:
   - Use <h3> for main section headings
   - Use <h4> for subsection headings
   - Use <h5> for minor headings

5. **Special Formatting**:
   - Use <code> for inline code, commands, or technical terms
   - Use <pre> for code blocks or formatted text that needs spacing preserved
   - Use <blockquote> for quoted text or important callouts
   - Use <mark> to highlight very important information

6. **Professional Elements**:
   - Use <div class="alert"> or <div class="note"> style blocks for warnings/notes
   - Use <span> with inline styles for colored text when emphasizing status (e.g., <span style="color: green;">✓ Success</span>)
   - Use <table>, <tr>, <td> for tabular data when comparing information

EXAMPLE FORMAT (no \n characters):
<h3>Search Results</h3>
<p>Based on the knowledge base search, here's what I found:</p>
<ul>
  <li><strong>First Finding:</strong> Exact information from the search results</li>
  <li><strong>Second Finding:</strong> Another piece of information from the results</li>
</ul>
<blockquote>
  <p><strong>Note:</strong> This information comes directly from the knowledge base</p>
</blockquote>

TOOL USAGE GUIDELINES:
1. Use the search tool when users ask about specific procedures, policies, documentation, or factual information
2. Choose the appropriate context (Crew, Developer, or PolicyExpert) based on the nature of the query
3. After retrieving passages, synthesize ONLY the information found - do not add external knowledge
4. If no relevant information is found, acknowledge this clearly and do not attempt to answer from general knowledge

RESPONSE GUIDELINES:
- Always be transparent about what was found or not found in searches
- Use proper HTML structure throughout
- Make responses scannable with good visual hierarchy
- Bold key terms and important information
- Let HTML tags handle spacing - never use \n or \n\n
- Include relevant links when referencing sources
- Be professional, clear, and helpful in all responses

Remember: 
- ALWAYS use HTML formatting. Never use markdown (* or ** or # or -)
- Never use \n or \n\n for spacing
- NEVER make up or add information beyond what the search tools return
- Your credibility depends on accuracy - only state what you can verify from tool results"""
            }
        ]
    
    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize Coveo search tool only"""
        return [
            {
                "toolSpec": {
                    "name": "search_passages",
                    "description": "Search and retrieve relevant passages from the Coveo knowledge base. Use this when users ask questions that require specific information from documentation, policies, or internal knowledge bases.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query or question to find relevant passages"
                                },
                                "context": {
                                    "type": "string",
                                    "description": "The context/tab for the search. Choose based on the user's role or query type",
                                    "enum": ["Crew", "Developer", "PolicyExpert"]
                                },
                                "maxPassages": {
                                    "type": "integer",
                                    "description": "Maximum number of passages to retrieve",
                                    "default": 5,
                                    "minimum": 1,
                                    "maximum": 10
                                }
                            },
                            "required": ["query", "context"]
                        }
                    }
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any], user_email: str = None) -> str:
        """Execute the Coveo search tool"""
        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
        
        if tool_name == "search_passages":
            return await self._search_coveo_passages(
                tool_input.get("query", ""),
                tool_input.get("context", "Crew"),
                tool_input.get("maxPassages", 5),
                user_email
            )
        else:
            return f"Unknown tool: {tool_name}"
    
    async def _get_coveo_token(self, user_email: str) -> Optional[str]:
        """
        Get a Coveo search token for the user (cached or new)
        
        Args:
            user_email: The email of the user making the request
            
        Returns:
            The search token or None if failed
        """
        try:
            # Check cache first
            cached_data = await self.token_cache.get_cached_token(user_email)
            if cached_data:
                return cached_data["token"]
            
            # No valid cached token, generate a new one
            logger.info(f"Generating new Coveo token for user: {user_email}")
            
            coveo_org_id = config.COVEO_ORG_ID
            coveo_api_key = config.COVEO_API_KEY
            
            if not coveo_org_id or not coveo_api_key:
                logger.warning("Coveo organization ID or API key not configured")
                return None
            
            # Construct the Coveo token endpoint with organization ID and API key
            url = f"https://platform.cloud.coveo.com/rest/search/v2/token?organizationId={coveo_org_id}"
            
            # Set token validity to 10 minutes
            valid_for_ms = 600000  # 10 minutes in milliseconds
            
            # Prepare the request payload
            payload = {
                "userIds": [
                    {
                        "name": user_email,
                        "provider": "Email Security Provider"
                    }
                ],
                "validFor": valid_for_ms
            }
            
            # Use the API key directly in the Authorization header
            headers = {
                "Authorization": f"Bearer {coveo_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            logger.info(f"Requesting new Coveo token for user: {user_email} (valid for 10 minutes)")
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        token = data.get("token")
                        if token:
                            # Cache the token
                            await self.token_cache.set_token(user_email, token, valid_for_ms)
                            logger.info(f"Successfully obtained and cached Coveo token for user: {user_email}")
                            return token
                        else:
                            logger.error("No token in Coveo response")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to get Coveo token. Status: {response.status}, Error: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting Coveo token: {str(e)}")
            # Try to invalidate cache for this user in case of errors
            async with self.token_cache.lock:
                if user_email in self.token_cache.cache:
                    del self.token_cache.cache[user_email]
                    logger.info(f"Removed potentially invalid token for {user_email} from cache")
            return None
    
    async def _search_coveo_passages(self, query: str, context: str, max_passages: int = 5, user_email: str = None) -> str:
        """
        Search Coveo Passage Retrieval API for relevant passages
        
        Context definitions:
        - Crew: General crew member questions, operational procedures, daily tasks
        - Developer: Technical questions, API documentation, coding guidelines, development tools
        - PolicyExpert: Policy-related questions, compliance, regulations, governance
        """
        try:
            # Get Coveo configuration from environment
            coveo_org_id = config.COVEO_ORG_ID
            search_hub = config.COVEO_SEARCH_HUB
            
            if not coveo_org_id:
                logger.warning("Coveo organization ID not configured")
                return "<p style='color: orange;'>I couldn't search the knowledge base as the Coveo integration is not configured.</p>"
            
            # Get a token for this user (cached or new)
            if not user_email:
                logger.warning("No user email provided for Coveo token")
                return "<p style='color: orange;'>Unable to authenticate for knowledge base search.</p>"
            
            coveo_token = await self._get_coveo_token(user_email)
            if not coveo_token:
                logger.error("Failed to obtain Coveo token")
                return "<p style='color: red;'>Failed to authenticate with the knowledge base.</p>"
            
            # Construct the Coveo Passage Retrieval API endpoint
            url = f"https://{coveo_org_id}.org.coveo.com/rest/search/v3/passages/retrieve"
            
            # Prepare the request payload
            payload = {
                "query": query,
                "additionalFields": [
                    "clickableuri",
                    "title"
                ],
                "maxPassages": max_passages,
                "searchHub": search_hub,
                "localization": {
                    "locale": "en-US",
                    "timezone": "America/New_York"
                },
                "context": {
                    "tab": context
                }
            }
            
            # Use the user-specific token for the search
            headers = {
                "Authorization": f"Bearer {coveo_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            logger.info(f"Searching Coveo with query: {query} for user: {user_email}")
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format the passages for the model with HTML formatting
                        if "items" in data and data["items"]:
                            passages_text = f"<h4>Search Results from {context} Knowledge Base</h4>\n"
                            passages_text += f"<p><em>Query: \"{query}\"</em></p>\n<hr>\n\n"
                            
                            for i, item in enumerate(data["items"], 1):
                                passages_text += f"<div class='passage'>\n"
                                passages_text += f"<h5>Passage {i}</h5>\n"
                                passages_text += f"<blockquote>{item.get('text', 'N/A')}</blockquote>\n"
                                
                                if "document" in item:
                                    doc = item["document"]
                                    passages_text += "<p class='source-info'>"
                                    if "title" in doc:
                                        passages_text += f"<strong>Source:</strong> {doc['title']}<br>"
                                    if "clickableuri" in doc:
                                        passages_text += f"<strong>Link:</strong> <a href='{doc['clickableuri']}'>{doc['clickableuri']}</a><br>"
                                    passages_text += "</p>"
                                
                                if "relevanceScore" in item:
                                    relevance_pct = item['relevanceScore'] * 100
                                    color = "green" if relevance_pct > 80 else "orange" if relevance_pct > 60 else "gray"
                                    passages_text += f"<p><span style='color: {color};'>Relevance: {relevance_pct:.1f}%</span></p>\n"
                                
                                passages_text += "</div>\n"
                                if i < len(data["items"]):
                                    passages_text += "<hr style='margin: 20px 0; border: 0; border-top: 1px solid #eee;'>\n"
                            
                            logger.info(f"Found {len(data['items'])} passages for query: {query}")
                            return passages_text
                        else:
                            return f"<p><em>No relevant passages found in the {context} knowledge base for '{query}'.</em></p>"
                    elif response.status == 401:
                        # Token might be invalid, try to clear cache and retry once
                        logger.warning(f"Authentication failed for user {user_email}, clearing cache")
                        async with self.token_cache.lock:
                            if user_email in self.token_cache.cache:
                                del self.token_cache.cache[user_email]
                        
                        # Try one more time with a fresh token
                        logger.info("Retrying with fresh token...")
                        fresh_token = await self._get_coveo_token(user_email)
                        if fresh_token:
                            headers["Authorization"] = f"Bearer {fresh_token}"
                            async with session.post(url, json=payload, headers=headers) as retry_response:
                                if retry_response.status == 200:
                                    data = await retry_response.json()
                                    # Process the response (same as above)
                                    if "items" in data and data["items"]:
                                        # [Same processing logic as above - omitted for brevity]
                                        logger.info(f"Retry successful, found {len(data['items'])} passages")
                                        return self._format_search_results(data, context, query)
                        
                        return f"<p style='color: red;'>Authentication failed with the knowledge base.</p>"
                    else:
                        error_text = await response.text()
                        logger.error(f"Coveo API error: {response.status} - {error_text}")
                        return f"<p style='color: red;'>I encountered an error searching the knowledge base (Status: {response.status})</p>"
                        
        except Exception as e:
            logger.error(f"Error calling Coveo Passage Retrieval API: {str(e)}")
            return f"<p style='color: red;'>I encountered an error while searching the knowledge base: {str(e)}</p>"
    
    def _format_search_results(self, data: Dict, context: str, query: str) -> str:
        """Helper method to format search results"""
        if "items" in data and data["items"]:
            passages_text = f"<h4>Search Results from {context} Knowledge Base</h4>\n"
            passages_text += f"<p><em>Query: \"{query}\"</em></p>\n<hr>\n\n"
            
            for i, item in enumerate(data["items"], 1):
                passages_text += f"<div class='passage'>\n"
                passages_text += f"<h5>Passage {i}</h5>\n"
                passages_text += f"<blockquote>{item.get('text', 'N/A')}</blockquote>\n"
                
                if "document" in item:
                    doc = item["document"]
                    passages_text += "<p class='source-info'>"
                    if "title" in doc:
                        passages_text += f"<strong>Source:</strong> {doc['title']}<br>"
                    if "clickableuri" in doc:
                        passages_text += f"<strong>Link:</strong> <a href='{doc['clickableuri']}'>{doc['clickableuri']}</a><br>"
                    passages_text += "</p>"
                
                if "relevanceScore" in item:
                    relevance_pct = item['relevanceScore'] * 100
                    color = "green" if relevance_pct > 80 else "orange" if relevance_pct > 60 else "gray"
                    passages_text += f"<p><span style='color: {color};'>Relevance: {relevance_pct:.1f}%</span></p>\n"
                
                passages_text += "</div>\n"
                if i < len(data["items"]):
                    passages_text += "<hr style='margin: 20px 0; border: 0; border-top: 1px solid #eee;'>\n"
            
            return passages_text
        else:
            return f"<p><em>No relevant passages found in the {context} knowledge base for '{query}'.</em></p>"
            
            # Make the API request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format the passages for the model with HTML formatting
                        if "items" in data and data["items"]:
                            passages_text = f"<h4>Search Results from {context} Knowledge Base</h4>\n"
                            passages_text += f"<p><em>Query: \"{query}\"</em></p>\n<hr>\n\n"
                            
                            for i, item in enumerate(data["items"], 1):
                                passages_text += f"<div class='passage'>\n"
                                passages_text += f"<h5>Passage {i}</h5>\n"
                                passages_text += f"<blockquote>{item.get('text', 'N/A')}</blockquote>\n"
                                
                                if "document" in item:
                                    doc = item["document"]
                                    passages_text += "<p class='source-info'>"
                                    if "title" in doc:
                                        passages_text += f"<strong>Source:</strong> {doc['title']}<br>"
                                    if "clickableuri" in doc:
                                        passages_text += f"<strong>Link:</strong> <a href='{doc['clickableuri']}'>{doc['clickableuri']}</a><br>"
                                    passages_text += "</p>"
                                
                                if "relevanceScore" in item:
                                    relevance_pct = item['relevanceScore'] * 100
                                    color = "green" if relevance_pct > 80 else "orange" if relevance_pct > 60 else "gray"
                                    passages_text += f"<p><span style='color: {color};'>Relevance: {relevance_pct:.1f}%</span></p>\n"
                                
                                passages_text += "</div>\n"
                                if i < len(data["items"]):
                                    passages_text += "<hr style='margin: 20px 0; border: 0; border-top: 1px solid #eee;'>\n"
                            
                            logger.info(f"Found {len(data['items'])} passages for query: {query}")
                            return passages_text
                        else:
                            return f"<p><em>No relevant passages found in the {context} knowledge base for '{query}'.</em></p>"
                    else:
                        error_text = await response.text()
                        logger.error(f"Coveo API error: {response.status} - {error_text}")
                        return f"<p style='color: red;'>I encountered an error searching the knowledge base (Status: {response.status})</p>"
                        
        except Exception as e:
            logger.error(f"Error calling Coveo Passage Retrieval API: {str(e)}")
            return f"<p style='color: red;'>I encountered an error while searching the knowledge base: {str(e)}</p>"
    
    def build_message_history(self, conversation_history: List[Message], new_message: str) -> List[dict]:
        """Build message history for Bedrock API"""
        messages = []
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({
                "role": msg.role,
                "content": [{"text": msg.content}]
            })
        
        # Add new user message
        messages.append({
            "role": "user",
            "content": [{"text": new_message}]
        })
        
        return messages
    
    def sync_converse_stream(
        self, 
        messages: List[dict], 
        max_tokens: int, 
        temperature: float,
        model_type: str,
        use_tools: bool = False,
        system_prompt: Optional[List[Dict[str, Any]]] = None
    ):
        """Synchronous call to Bedrock converse_stream API with optional tool support"""
        client = self.bedrock_manager.get_client()
        model_id = self.bedrock_manager.get_model_id(model_type)
        
        logger.info(f"Streaming with model: {model_id} (type: {model_type}, tools: {use_tools})")
        
        request_params = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_params["system"] = system_prompt
        
        # Add tools if enabled
        if use_tools:
            request_params["toolConfig"] = {
                "tools": self.tools
            }
        
        return client.converse_stream(**request_params)
    
    def _stream_worker_with_tools(
        self, 
        messages: List[dict], 
        max_tokens: int,
        temperature: float,
        model_type: str,
        result_queue: queue.Queue,
        user_email: str = None
    ):
        """Worker thread for streaming responses with tool support"""
        try:
            # Track conversation for tool use
            current_messages = messages.copy()
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Iteration {iteration}: Making Bedrock call with {len(current_messages)} messages")
                
                # Make the streaming call with system prompt
                response = self.sync_converse_stream(
                    current_messages, 
                    max_tokens, 
                    temperature, 
                    model_type,
                    use_tools=True,
                    system_prompt=self.system_prompt
                )
                
                tool_use_detected = False
                tool_use_id = None
                tool_name = None
                tool_input = ""
                accumulated_text = ""
                
                # Process the stream
                for event in response.get("stream", []):
                    if "contentBlockStart" in event:
                        block = event["contentBlockStart"]["start"]
                        if "toolUse" in block:
                            tool_use_detected = True
                            tool_use_id = block["toolUse"]["toolUseId"]
                            tool_name = block["toolUse"]["name"]
                            logger.info(f"Tool use detected: {tool_name} (ID: {tool_use_id})")
                            
                            # Optionally notify client that search is happening
                            # result_queue.put({
                            #     "type": "text",
                            #     "content": "\n*Searching knowledge base...*\n"
                            # })
                    
                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        if "text" in delta:
                            # Send text chunk to client
                            result_queue.put({
                                "type": "text",
                                "content": delta["text"]
                            })
                            accumulated_text += delta["text"]
                        elif "toolUse" in delta:
                            # Accumulate tool input
                            if "input" in delta["toolUse"]:
                                tool_input += delta["toolUse"]["input"]
                    
                    elif "messageStop" in event:
                        logger.info(f"Message stop. Tool detected: {tool_use_detected}")
                        break
                
                # After the stream completes, handle tool execution if needed
                if tool_use_detected and tool_input:
                    try:
                        # Parse and execute the tool
                        tool_input_json = json.loads(tool_input)
                        logger.info(f"Executing {tool_name} with input: {tool_input_json}")
                        
                        # Execute tool synchronously with user email
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            tool_result = loop.run_until_complete(
                                self.execute_tool(tool_name, tool_input_json, user_email)
                            )
                        finally:
                            loop.close()
                        
                        logger.info(f"Tool execution complete. Result preview: {tool_result[:200]}...")
                        
                        # Build assistant message with tool use
                        assistant_content = []
                        if accumulated_text:
                            assistant_content.append({"text": accumulated_text})
                        assistant_content.append({
                            "toolUse": {
                                "toolUseId": tool_use_id,
                                "name": tool_name,
                                "input": tool_input_json
                            }
                        })
                        
                        # Add assistant message to conversation
                        current_messages.append({
                            "role": "assistant",
                            "content": assistant_content
                        })
                        
                        # Add tool result as user message
                        current_messages.append({
                            "role": "user",
                            "content": [{
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": [{"text": tool_result}]
                                }
                            }]
                        })
                        
                        logger.info(f"Tool result added. Continuing to next iteration for final response...")
                        # Continue to next iteration to get Bedrock's response with the tool results
                        continue
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool input: {e}")
                        result_queue.put({
                            "type": "error",
                            "content": f"Failed to parse tool input: {e}"
                        })
                        return
                    except Exception as e:
                        logger.error(f"Tool execution error: {e}")
                        result_queue.put({
                            "type": "error",
                            "content": f"Tool execution error: {e}"
                        })
                        return
                else:
                    # No tool was used, we're done
                    logger.info("No tool use detected, streaming complete")
                    result_queue.put({"type": "done"})
                    return
            
            # Safety: if we exit the loop due to max iterations
            logger.warning(f"Reached maximum iterations ({max_iterations})")
            result_queue.put({"type": "done"})
                
        except Exception as e:
            logger.error(f"Stream worker error: {str(e)}", exc_info=True)
            result_queue.put({
                "type": "error",
                "content": str(e)
            })
    
    def _stream_worker_simple(
        self, 
        messages: List[dict], 
        max_tokens: int,
        temperature: float,
        model_type: str,
        result_queue: queue.Queue
    ):
        """Simple worker thread for streaming responses without tools"""
        try:
            response = self.sync_converse_stream(
                messages, 
                max_tokens, 
                temperature, 
                model_type,
                use_tools=False,
                system_prompt=self.system_prompt
            )
            
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        # Send only the text content
                        result_queue.put({
                            "type": "text",
                            "content": delta["text"]
                        })
                elif "messageStop" in event:
                    # Signal completion
                    result_queue.put({"type": "done"})
                    break
                    
            # Ensure done is sent if not already sent
            result_queue.put({"type": "done"})
            
        except Exception as e:
            logger.error(f"Stream worker error: {str(e)}")
            result_queue.put({
                "type": "error",
                "content": str(e)
            })
    
    async def converse_stream_async(
        self, 
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """Async generator for streaming responses in SSE format with tool support"""
        try:
            messages = self.build_message_history(
                request.conversation_history,
                request.prompt
            )
            
            max_tokens = request.max_tokens or config.DEFAULT_MAX_TOKENS
            temperature = request.temperature or config.DEFAULT_TEMPERATURE
            
            # Create queue for thread communication
            result_queue = queue.Queue()
            
            # Determine if tools should be used based on request parameter and prompt content
            should_use_tools = request.use_tools and (
                # Check for knowledge base queries
                any(keyword in request.prompt.lower() 
                    for keyword in ["how", "what", "when", "where", "why", "explain", "tell me about", 
                                  "describe", "guide", "help", "procedure", "policy", "documentation",
                                  "search", "find", "look up", "retrieve", "information about"])
            )
            
            logger.info(f"Starting stream for user {request.email}. Tools enabled: {should_use_tools}")
            
            # Use appropriate worker based on tool usage
            if should_use_tools:
                thread = threading.Thread(
                    target=self._stream_worker_with_tools,
                    args=(
                        messages,
                        max_tokens,
                        temperature,
                        request.bedrockModelType,
                        result_queue,
                        request.email  # Pass user email for Coveo token
                    ),
                    daemon=True
                )
            else:
                # Use simple worker without tools
                thread = threading.Thread(
                    target=self._stream_worker_simple,
                    args=(
                        messages,
                        max_tokens,
                        temperature,
                        request.bedrockModelType,
                        result_queue
                    ),
                    daemon=True
                )
            
            thread.start()
            
            # Process results as they come in
            while True:
                try:
                    event = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: result_queue.get(timeout=0.1)
                    )
                    
                    # Format SSE response based on event type
                    if event["type"] == "text":
                        # Escape the content for JSON
                        escaped_content = json.dumps(event["content"])[1:-1]  # Remove quotes
                        yield f'data: {{"text": "{escaped_content}"}}\n\n'
                    elif event["type"] == "done":
                        # Send completion signal
                        yield 'data: {"done": true}\n\n'
                        break
                    elif event["type"] == "error":
                        # Send error in SSE format
                        escaped_error = json.dumps(event["content"])[1:-1]
                        yield f'data: {{"error": "{escaped_error}"}}\n\n'
                        break
                        
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"Error processing queue event: {e}")
                    yield f'data: {{"error": "Stream processing error"}}\n\n'
                    break
                    
        except Exception as e:
            logger.error(f"Stream async error: {str(e)}")
            yield f'data: {{"error": "{str(e)}"}}\n\n'
