# Architecture Overview

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                            │
│  ┌──────────────┐              ┌──────────────┐             │
│  │  Dashboard   │              │   REST API   │             │
│  │  (Streamlit) │              │  (FastAPI)   │             │
│  └──────────────┘              └──────────────┘             │
└─────────────────┬───────────────────────┬────────────────────┘
                  │                       │
┌─────────────────▼───────────────────────▼────────────────────┐
│                  APPLICATION LAYER                           │
│  ┌──────────────────────────────────────────────────────┐    │
│  │            Inference Engine                         │    │
│  │  - Model Loading & Versioning                       │    │
│  │  - Batch & Real-time Predictions                    │    │
│  │  - Uncertainty Quantification (SHAP/LIME)           │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────┬───────────────────────┬────────────────────┘
                  │                       │
┌─────────────────▼───────────────────────▼────────────────────┐
│                   ML/DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Feature    │  │   Training   │  │    Model     │        │
│  │  Engineering │  │   Pipeline   │  │  Registry    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────┬───────────────────────┬────────────────────┘
                  │                       │
┌─────────────────▼───────────────────────▼────────────────────┐
│                  DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Database   │  │    Cache     │  │   Vector DB  │        │
│  │ (PostgreSQL) │  │   (Redis)    │  │  (Faiss/PG) │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/
├── api/              # FastAPI application
│   ├── main.py       # App initialization & endpoints
│   ├── dependencies.py
│   └── routers/
│       └── predictions.py
├── core/             # Core utilities
│   ├── exceptions.py
│   ├── logging.py
│   └── database.py
├── schemas/          # Pydantic models
│   ├── base.py
│   └── predictions.py
├── models/           # ML model implementations
│   ├── base.py
│   └── ensemble.py
├── data/             # Data processing
│   ├── loaders.py
│   ├── validation.py
│   └── preprocessing.py
├── inference/        # Prediction engine
│   ├── engine.py
│   └── explainability.py
└── dashboard/        # Streamlit UI
    └── app.py

projects/
├── finsight/         # Financial forecasting project
│   ├── config/
│   ├── src/
│   ├── notebooks/
│   └── tests/
└── [other projects...]
```

## Key Design Principles

### 1. **Strict Separation of Concerns**
- **Data Layer**: ETL, validation, storage
- **Model Layer**: Training, versioning, registry
- **Inference Layer**: Predictions, caching, monitoring
- **API Layer**: REST endpoints, request/response handling
- **UI Layer**: Dashboard, visualization

### 2. **Type Safety**
- Pydantic v2 for all schemas
- Strict type hints throughout
- Mypy validation in CI/CD

### 3. **Error Handling**
- Custom exception hierarchy
- Graceful degradation
- Proper HTTP status codes

### 4. **Monitoring & Observability**
- Structured logging (JSON format)
- MLflow experiment tracking
- Prometheus metrics
- Alerts & dashboards

### 5. **Production Readiness**
- Database migrations (Alembic)
- Model versioning (MLflow registry)
- Health checks & liveness probes
- Container orchestration ready

## Data Flow: Finsight Example

```
1. DATA INGESTION
   External APIs → Data Loader → Polars DataFrame
                                      ↓
2. VALIDATION
   Schema Validation → Data Quality Checks → Store in DB
                                      ↓
3. FEATURE ENGINEERING
   Raw Features → Transformations → Feature Store (PostgreSQL)
                                      ↓
4. MODEL TRAINING
   Historical Data → Ensemble Training → Model Registry (MLflow)
                                      ↓
5. INFERENCE
   API Request → Load Model → Predict → SHAP Explain → Cache Response
                                      ↓
6. API RESPONSE
   Prediction + Confidence + Explanation → JSON Response
                                      ↓
7. VISUALIZATION
   API → Dashboard (Streamlit) → Interactive Charts (Plotly)
```

## Technology Justification

| Component | Technology | Why |
|-----------|-----------|-----|
| API Framework | FastAPI | Auto-docs, type hints, async support, speed |
| Data Processing | Polars | Performance, type safety, better than Pandas |
| Statistical Modeling | PyMC3 | Bayesian inference, posterior sampling, uncertainty |
| ML Ensemble | LightGBM + XGBoost | Speed, interpretability, gradient boosting |
| Database | PostgreSQL | ACID, JSON support, reliable |
| Caching | Redis | Fast, distributed, session management |
| Monitoring | MLflow | Industry standard, model registry, experiment tracking |
| Dashboard | Streamlit | Fast development, interactive, deployment-ready |
| Containerization | Docker | Reproducibility, scaling, CI/CD integration |
| Orchestration | Kubernetes | Auto-scaling, self-healing, production-grade |

## Deployment Strategy

### Development
```
Local Dev Environment
├── Docker Compose (all services)
├── Hot-reload
└── Test Database
```

### Staging
```
Kubernetes Cluster (staging namespace)
├── PostgreSQL (managed)
├── Redis (managed)
├── API (1 replica)
├── Dashboard (1 replica)
└── MLflow (1 replica)
```

### Production
```
Kubernetes Cluster (production namespace)
├── PostgreSQL (highly available, backups)
├── Redis (cluster, sentinel)
├── API (3+ replicas, auto-scaling)
├── Dashboard (2+ replicas)
├── MLflow (2+ replicas)
├── Prometheus (monitoring)
├── ELK Stack (logging)
└── Load Balancer (SSL/TLS)
```

## Error Handling Strategy

1. **Validation Errors (422)**: Invalid input data
2. **Not Found Errors (404)**: Resource doesn't exist
3. **Inference Errors (500)**: Model prediction failed
4. **Rate Limit (429)**: Too many requests
5. **Server Errors (500)**: Unexpected failures

All errors include:
- Machine-readable error code
- Human-readable message
- Request ID for tracking
- Timestamp

## Monitoring & Alerting

### Metrics
- API response time (p50, p95, p99)
- Model prediction latency
- Cache hit rate
- Database connection pool
- Error rates by type

### Alerts
- API response time > 1s (warning)
- Error rate > 1% (critical)
- Database unavailable (critical)
- Model drift detected (warning)
- Low disk space (warning)

## Security Considerations

1. **Database**: Encrypted credentials, connection pooling
2. **API**: HTTPS only, CORS configuration, rate limiting
3. **Secrets**: Environment variables, secret manager
4. **Logging**: No sensitive data in logs
5. **Access**: API key authentication, role-based access control
