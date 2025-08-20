import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { BedrockRuntimeClient, InvokeModelWithResponseStreamCommand } from '@aws-sdk/client-bedrock-runtime';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 8001;
const NODE_ENV = process.env.NODE_ENV || 'development';

// Security middleware
app.use(helmet({
  contentSecurityPolicy: false,
  crossOriginEmbedderPolicy: false
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: NODE_ENV === 'production' ? 100 : 1000, // limit each IP to 100 requests per windowMs in production
  message: 'Too many requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/chat', limiter);

// CORS configuration
const corsOptions = {
  origin: NODE_ENV === 'production' 
    ? process.env.ALLOWED_ORIGINS?.split(',') || []
    : ['http://localhost:5173', 'http://127.0.0.1:5173'],
  credentials: true,
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
};

app.use(cors(corsOptions));
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true, limit: '1mb' }));

// Enhanced logging middleware
app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  const ip = req.ip || req.connection.remoteAddress;
  console.log(`${timestamp} - ${req.method} ${req.path} - IP: ${ip}`);
  next();
});

// AWS Bedrock client configuration
const clientConfig = {
  region: process.env.AWS_REGION || 'us-east-1'
};

if (process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY) {
  clientConfig.credentials = {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  };
}

const bedrockClient = new BedrockRuntimeClient(clientConfig);

class BedrockService {
  constructor() {
    this.modelId = process.env.BEDROCK_MODEL_ID || 'anthropic.claude-3-sonnet-20240229-v1:0';
    this.region = process.env.AWS_REGION || 'us-east-1';
    this.maxTokens = parseInt(process.env.MAX_TOKENS) || 4000;
    this.temperature = parseFloat(process.env.TEMPERATURE) || 0.7;
  }

  validateMessages(messages) {
    if (!Array.isArray(messages)) {
      throw new Error('Messages must be an array');
    }

    if (messages.length === 0) {
      throw new Error('Messages array cannot be empty');
    }

    const validRoles = ['user', 'assistant', 'system'];
    for (const msg of messages) {
      if (!msg.role || !validRoles.includes(msg.role)) {
        throw new Error(`Invalid message role: ${msg.role}`);
      }
      if (!msg.content || typeof msg.content !== 'string') {
        throw new Error('Message content must be a non-empty string');
      }
      if (msg.content.length > 50000) {
        throw new Error('Message content too long');
      }
    }

    return true;
  }

  formatMessagesForClaude(messages) {
    let systemMessage = '';
    const formattedMessages = [];

    messages.forEach(msg => {
      if (msg.role === 'system') {
        systemMessage = msg.content;
      } else if (['user', 'assistant'].includes(msg.role)) {
        formattedMessages.push({
          role: msg.role,
          content: msg.content
        });
      }
    });

    const body = {
      anthropic_version: 'bedrock-2023-05-31',
      max_tokens: this.maxTokens,
      messages: formattedMessages,
      temperature: this.temperature
    };

    if (systemMessage) {
      body.system = systemMessage;
    }

    return body;
  }

  async *streamChat(request) {
    try {
      this.validateMessages(request.messages);

      const body = this.formatMessagesForClaude(request.messages);
      
      // Override defaults with request parameters if provided
      if (request.max_tokens && request.max_tokens <= 8000) {
        body.max_tokens = request.max_tokens;
      }
      if (request.temperature !== undefined && request.temperature >= 0 && request.temperature <= 1) {
        body.temperature = request.temperature;
      }

      console.log(`Sending request to Bedrock model: ${this.modelId}`);

      const command = new InvokeModelWithResponseStreamCommand({
        modelId: this.modelId,
        body: JSON.stringify(body)
      });

      const response = await bedrockClient.send(command);

      for await (const event of response.body) {
        if (event.chunk?.bytes) {
          const chunkData = JSON.parse(Buffer.from(event.chunk.bytes).toString());

          if (chunkData.type === 'content_block_delta') {
            const delta = chunkData.delta || {};
            if (delta.type === 'text_delta') {
              const text = delta.text || '';
              if (text) {
                yield JSON.stringify({ content: text });
                await new Promise(resolve => setTimeout(resolve, 5));
              }
            }
          } else if (chunkData.type === 'message_stop') {
            console.log('Message streaming completed');
            yield JSON.stringify({ done: true });
            break;
          } else if (chunkData.type === 'error') {
            throw new Error(chunkData.error?.message || 'Unknown error from Bedrock');
          }
        }
      }
    } catch (error) {
      console.error('Error in streamChat:', error);
      yield JSON.stringify({ 
        error: error.message || 'An error occurred while processing your request',
        type: 'error'
      });
    }
  }
}

// Initialize service
const bedrockService = new BedrockService();

// Input validation middleware
const validateChatRequest = (req, res, next) => {
  try {
    const { messages, max_tokens, temperature } = req.body;

    if (!messages) {
      return res.status(400).json({ error: 'Messages field is required' });
    }

    if (max_tokens !== undefined && (typeof max_tokens !== 'number' || max_tokens < 1 || max_tokens > 8000)) {
      return res.status(400).json({ error: 'max_tokens must be a number between 1 and 8000' });
    }

    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 1)) {
      return res.status(400).json({ error: 'temperature must be a number between 0 and 1' });
    }

    bedrockService.validateMessages(messages);
    next();
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
};

// Routes
app.post('/chat/stream', validateChatRequest, async (req, res) => {
  try {
    const { messages, max_tokens, temperature } = req.body;

    // Set SSE headers
    res.writeHead(200, {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': corsOptions.origin.includes(req.headers.origin) ? req.headers.origin : corsOptions.origin[0],
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    });

    res.write('data: {"status": "started"}\n\n');

    const request = { messages, max_tokens, temperature };

    try {
      for await (const chunk of bedrockService.streamChat(request)) {
        res.write(`data: ${chunk}\n\n`);
      }
    } catch (streamError) {
      console.error('Streaming error:', streamError);
      res.write(`data: ${JSON.stringify({ 
        error: 'Streaming interrupted', 
        type: 'stream_error' 
      })}\n\n`);
    }

    res.write('data: [DONE]\n\n');
    res.end();

  } catch (error) {
    console.error('Error in /chat/stream:', error);
    if (!res.headersSent) {
      res.status(500).json({ 
        error: NODE_ENV === 'production' ? 'Internal server error' : error.message 
      });
    }
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'bedrock-chat-api',
    model: bedrockService.modelId,
    environment: NODE_ENV,
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// API info endpoint
app.get('/api/info', (req, res) => {
  res.json({
    service: 'Bedrock Chat API',
    version: '1.0.0',
    model: bedrockService.modelId,
    endpoints: [
      {
        path: '/chat/stream',
        method: 'POST',
        description: 'Stream chat responses from Claude'
      },
      {
        path: '/health',
        method: 'GET',
        description: 'Health check endpoint'
      }
    ]
  });
});

// Global error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  const message = NODE_ENV === 'production' ? 'Internal server error' : error.message;
  res.status(500).json({ error: message });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  process.exit(0);
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`🚀 Bedrock Chat API running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`💬 Chat endpoint: http://localhost:${PORT}/chat/stream`);
  console.log(`🔧 Environment: ${NODE_ENV}`);
  console.log(`🤖 Model: ${bedrockService.modelId}`);
});

// Handle server errors
server.on('error', (error) => {
  console.error('Server error:', error);
  process.exit(1);
});

export default app;
