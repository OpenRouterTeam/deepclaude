use crate::config::OpenRouterConfig;
use crate::error::{ApiError, Result};
use crate::models::request::{ChatRequest, ChatResponse, Message, Model};
use openai::v1::{
    api::Client as OpenAIClient,
    chat_completion::{ChatCompletionMessage, ChatCompletionRequest, Role},
};
use reqwest::header::{HeaderMap, HeaderValue};

pub struct OpenRouterClient {
    client: OpenAIClient,
}

impl OpenRouterClient {
    pub fn new(config: &OpenRouterConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        if let Some(url) = &config.site_url {
            headers.insert(
                "HTTP-Referer",
                HeaderValue::from_str(url).map_err(|e| ApiError::BadRequest {
                    message: format!("Invalid site URL: {}", e),
                })?,
            );
        }
        if let Some(name) = &config.site_name {
            headers.insert(
                "X-Title",
                HeaderValue::from_str(name).map_err(|e| ApiError::BadRequest {
                    message: format!("Invalid site name: {}", e),
                })?,
            );
        }

        let client = OpenAIClient::new()
            .with_api_key(&config.api_key)
            .with_base_url("https://openrouter.ai/api/v1")
            .with_headers(headers)
            .build()
            .map_err(|e| ApiError::BadRequest {
                message: format!("Failed to create OpenRouter client: {}", e),
            })?;

        Ok(Self { client })
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let model_id = match request.model {
            Model::Claude => "anthropic/claude-2.1",
            Model::DeepSeek => "deepseek/deepseek-chat",
        };

        let messages: Vec<ChatCompletionMessage> = request
            .messages
            .into_iter()
            .map(|msg| match msg {
                Message::User(content) => ChatCompletionMessage {
                    role: Role::User,
                    content,
                    name: None,
                },
                Message::Assistant(content) => ChatCompletionMessage {
                    role: Role::Assistant,
                    content,
                    name: None,
                },
                Message::System(content) => ChatCompletionMessage {
                    role: Role::System,
                    content,
                    name: None,
                },
            })
            .collect();

        let completion = self
            .client
            .chat()
            .create(ChatCompletionRequest {
                model: model_id.to_string(),
                messages,
                temperature: Some(request.temperature),
                max_tokens: Some(request.max_tokens),
                stream: Some(false),
                ..Default::default()
            })
            .await
            .map_err(|e| ApiError::ExternalApi {
                message: format!("OpenRouter API error: {}", e),
            })?;

        let response_message = completion
            .choices
            .get(0)
            .ok_or_else(|| ApiError::ExternalApi {
                message: "No completion choices returned".to_string(),
            })?
            .message
            .content
            .clone();

        Ok(ChatResponse {
            message: Message::Assistant(response_message),
        })
    }
}
