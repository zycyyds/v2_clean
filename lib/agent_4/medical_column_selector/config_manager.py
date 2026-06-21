"""配置读取器：统一加载 .env 和 config.json。"""
import json
import os
import re
from typing import Dict, Any, Optional

try:
    from config_loader import get_agent_config
except Exception:
    get_agent_config = None


class ConfigManager:
    """管理项目配置，避免主流程里到处散落读取逻辑。"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器。"""
        if config_path is None:
            current_dir = os.path.dirname(__file__)
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, 'config.json')

        self.config_path = config_path
        self.project_root = os.path.dirname(self.config_path)
        self._load_project_env()
        self._config = self._load_config()

    def _load_project_env(self) -> None:
        """把项目根目录下的 .env 读入当前进程环境。"""
        env_path = os.path.join(self.project_root, ".env")
        if not os.path.exists(env_path):
            return

        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value

    def _load_config(self) -> Dict[str, Any]:
        """读取 config.json，并解析其中的环境变量占位符。"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return self._resolve_env_placeholders(json.load(f))
        return {}

    def _resolve_env_placeholders(self, value: Any) -> Any:
        """递归替换 ${VAR_NAME} 形式的配置占位符。"""
        if isinstance(value, dict):
            return {k: self._resolve_env_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_env_placeholders(v) for v in value]
        if isinstance(value, str):
            match = re.fullmatch(r"\$\{([A-Z0-9_]+)\}", value)
            if match:
                return os.environ.get(match.group(1), value)
        return value

    def get_model_config(self) -> Dict[str, Any]:
        """返回模型配置块。"""
        return self._config.get("model_configs", {})

    def get_workflow_config(self) -> Dict[str, Any]:
        """返回工作流配置块。"""
        return self._config.get("workflow_configs", {})

    def get_default_model_name(self) -> str:
        """默认只返回 OpenAI 模型，避免旧配置把非 OpenAI 模型带入流程。"""
        agent_cfg = get_agent_config("agent_4") if get_agent_config else {}
        yaml_model = str(agent_cfg.get("model") or agent_cfg.get("model_name") or "").strip()
        if yaml_model:
            return yaml_model

        env_model = str(os.environ.get("MODEL_NAME", "")).strip()
        if env_model:
            if env_model.lower().startswith("openai/"):
                return env_model
            if "/" not in env_model:
                return f"openai/{env_model}"

        model_configs = self.get_model_config()
        openai_cfg = model_configs.get("openai_configs", {})
        openai_model = str(openai_cfg.get("model", "")).strip()
        if openai_model:
            return f"openai/{openai_model}"

        return "openai/gpt-4.1-mini"

    def get_llm_api_config(self) -> Dict[str, Any]:
        agent_cfg = get_agent_config("agent_4") if get_agent_config else {}
        return {
            "api_key": agent_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
            "base_url": agent_cfg.get("base_url") or agent_cfg.get("api_base") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "model": agent_cfg.get("model") or agent_cfg.get("model_name") or self.get_default_model_name(),
            "timeout": int(agent_cfg.get("timeout", os.environ.get("OPENAI_TIMEOUT", "120"))),
            "temperature": float(agent_cfg.get("temperature", 0.2)),
            "task_text": agent_cfg.get("task_text", ""),
        }

    def get_model_settings(self, model_provider: str) -> Dict[str, Any]:
        """返回指定 provider 的配置。"""
        model_configs = self.get_model_config()
        return model_configs.get(f"{model_provider}_configs", {})

    def reload_config(self):
        """重新加载配置文件。"""
        self._config = self._load_config()


_config_manager = None


def get_config_manager() -> ConfigManager:
    """返回全局单例，避免重复读取配置。"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_default_model_name() -> str:
    """供外部模块直接获取默认模型名。"""
    return get_config_manager().get_default_model_name()


def get_llm_api_config() -> Dict[str, Any]:
    """供外部模块获取统一 LLM API 配置。"""
    return get_config_manager().get_llm_api_config()
