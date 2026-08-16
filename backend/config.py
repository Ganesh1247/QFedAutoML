"""
[IMPLEMENTED] Application configuration settings management via pydantic-settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "QFedAutoML"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security & Authentication
    SECRET_KEY: str = "default-dev-secret-key-change-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Database
    DATABASE_URL: str = "sqlite:///./qfedautoml.db"

    # Federated Learning
    FL_SERVER_HOST: str = "0.0.0.0"
    FL_SERVER_PORT: int = 8080
    FL_MIN_CLIENTS: int = 2
    FL_DEFAULT_ROUNDS: int = 5

    # Quantum Simulators (Simulators only by default)
    QUANTUM_SIMULATOR_BACKEND: str = "qiskit_aer"
    QUANTUM_MAX_QUBITS: int = 16
    QUANTUM_HARDWARE_ENABLED: bool = False
    IBM_QUANTUM_API_TOKEN: str | None = None

    # Privacy & Differential Privacy
    DP_EPSILON_DEFAULT: float = 1.0
    DP_DELTA_DEFAULT: float = 1e-5
    DP_MAX_GRAD_NORM: float = 1.0
    ENABLE_THREAT_DETECTION: bool = True

    # Logging & Monitoring
    LOG_LEVEL: str = "INFO"
    ENABLE_PROMETHEUS_METRICS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
