# UptoSkills AI Assistant

A modern, scalable architecture for the UptoSkills platform featuring modular frontend applications and a dedicated backend service.

## Project Architecture

```
uptoskills-ai-assistant/
│
├── frontend/
│   ├── dashboard/       # Dashboard pages, metrics, and widgets
│   ├── assessment/      # Skill assessments and testing workflows
│   ├── internship/      # Internship portal and project tracking
│   └── ai-assistant/    # Intelligent conversational assistant
│       ├── components/  # Chat window, message bubbles, input controls
│       ├── services/    # API clients, LLM integration, response handlers
│       ├── hooks/       # Conversation state, streaming, session hooks
│       └── styles/      # AI assistant-specific CSS and styling
│
└── backend/             # Server-side APIs, database, and business logic
```

## Module Overview

- **`frontend/dashboard/`**: User dashboard, analytics, progress monitoring, and widgets.
- **`frontend/assessment/`**: Candidate assessments, skill evaluations, tests, and scorecards.
- **`frontend/internship/`**: Internship discovery, application pipelines, and project milestones.
- **`frontend/ai-assistant/`**: Feature-isolated AI assistant with modular components, services, hooks, and styles.
- **`backend/`**: Server-side APIs, business logic, authentication, and database services.
