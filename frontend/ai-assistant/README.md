# AI Assistant Module

Feature-isolated AI Assistant module with clear separation of concerns across components, services, hooks, and styles.

## Internal Architecture

```
ai-assistant/
├── components/  # Presentation UI (ChatWindow, MessageBubble, InputBox, etc.)
├── services/    # Network & API communication (LLM calls, stream handlers)
├── hooks/       # Custom React hooks (chat state, streaming, history)
└── styles/      # AI assistant-specific CSS/SCSS styling
```

## Guidelines
- Keep business logic out of UI components.
- State management and side effects should reside in `hooks/`.
- Backend/LLM interactions should reside in `services/`.
