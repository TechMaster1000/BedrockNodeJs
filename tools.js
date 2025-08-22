def _stream_worker_with_tools(
    self, 
    messages: List[dict], 
    max_tokens: int,
    temperature: float,
    model_type: str,
    result_queue: queue.Queue
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
            
            response = self.sync_converse_stream(
                current_messages, 
                max_tokens, 
                temperature, 
                model_type,
                use_tools=True
            )
            
            tool_use_detected = False
            tool_use_id = None
            tool_name = None
            tool_input = ""
            accumulated_text = ""
            message_complete = False
            
            for event in response.get("stream", []):
                if "contentBlockStart" in event:
                    block = event["contentBlockStart"]["start"]
                    if "toolUse" in block:
                        tool_use_detected = True
                        tool_use_id = block["toolUse"]["toolUseId"]
                        tool_name = block["toolUse"]["name"]
                        logger.info(f"Tool use detected: {tool_name} with ID: {tool_use_id}")
                        
                        # Send a notification that tool is being used
                        result_queue.put({
                            "type": "text",
                            "content": f"\n\n*Searching knowledge base...*\n\n"
                        })
                
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
                    message_complete = True
                    logger.info(f"Message stop received. Tool detected: {tool_use_detected}")
            
            # After streaming completes, handle tool execution if needed
            if tool_use_detected and tool_input:
                try:
                    # Parse and execute the tool
                    tool_input_json = json.loads(tool_input)
                    logger.info(f"Executing tool {tool_name} with input: {tool_input_json}")
                    
                    # Execute tool synchronously
                    import asyncio
                    loop = asyncio.new_event_loop()
                    tool_result = loop.run_until_complete(
                        self.execute_tool(tool_name, tool_input_json)
                    )
                    loop.close()
                    
                    logger.info(f"Tool execution complete. Result length: {len(tool_result)}")
                    
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
                    
                    logger.info(f"Added tool result to conversation. Total messages: {len(current_messages)}")
                    
                    # Continue to next iteration to process tool results
                    continue
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool input: {e}")
                    result_queue.put({
                        "type": "error",
                        "content": f"Failed to parse tool input: {e}"
                    })
                    break
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    result_queue.put({
                        "type": "error", 
                        "content": f"Tool execution error: {e}"
                    })
                    break
            
            # If no tool was used and message is complete, we're done
            if message_complete and not tool_use_detected:
                logger.info("No tool use detected, streaming complete")
                result_queue.put({"type": "done"})
                return
            
            # Safety check - if we somehow get here without tool use or completion
            if not tool_use_detected and not message_complete:
                logger.warning("Unexpected state: no tool use and no message completion")
                result_queue.put({"type": "done"})
                return
        
        # If we've exhausted iterations
        logger.warning(f"Reached maximum iterations ({max_iterations})")
        result_queue.put({"type": "done"})
                
    except Exception as e:
        logger.error(f"Stream worker error: {str(e)}")
        result_queue.put({
            "type": "error",
            "content": str(e)
        })
