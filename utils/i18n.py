# ============================================================
# utils/i18n.py — 国际化/多语言支持模块
# 支持动态语言切换，所有文本外置到 JSON 配置文件
# ============================================================
import json
import os
from pathlib import Path
from typing import Dict, Optional


class I18nManager:
    """
    国际化管理器
    
    使用方法:
        i18n = I18nManager()
        text = i18n.t("app.title")  # 获取翻译
        i18n.set_language("ja")     # 切换语言
    """
    
    # 支持的语言
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "zh": "中文",
        "ja": "日本語",
    }
    
    def __init__(self, locales_dir: Optional[str] = None, default_lang: str = "zh"):
        """
        Args:
            locales_dir: 语言文件目录，默认项目根目录/locales
            default_lang: 默认语言代码
        """
        if locales_dir is None:
            base_dir = Path(__file__).parent.parent
            self.locales_dir = base_dir / "locales"
        else:
            self.locales_dir = Path(locales_dir)
        
        self.default_lang = default_lang
        self.current_lang = default_lang
        self._translations: Dict[str, Dict] = {}
        
        # 加载所有语言文件
        self._load_all_translations()
    
    def _load_all_translations(self):
        """加载所有语言文件"""
        for lang_code in self.SUPPORTED_LANGUAGES.keys():
            file_path = self.locales_dir / f"{lang_code}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self._translations[lang_code] = json.load(f)
                except Exception as e:
                    print(f"[I18n] 加载语言文件失败 {lang_code}: {e}")
                    self._translations[lang_code] = {}
            else:
                print(f"[I18n] 语言文件不存在: {file_path}")
                self._translations[lang_code] = {}
    
    def set_language(self, lang_code: str):
        """
        设置当前语言
        
        Args:
            lang_code: 语言代码 (en/zh/ja)
        """
        if lang_code in self.SUPPORTED_LANGUAGES:
            self.current_lang = lang_code
        else:
            print(f"[I18n] 不支持的语言: {lang_code}, 使用默认语言")
            self.current_lang = self.default_lang
    
    def get_language(self) -> str:
        """获取当前语言代码"""
        return self.current_lang
    
    def get_language_name(self, lang_code: Optional[str] = None) -> str:
        """获取语言名称"""
        code = lang_code or self.current_lang
        return self.SUPPORTED_LANGUAGES.get(code, code)
    
    def t(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """
        获取翻译文本
        
        Args:
            key: 翻译键，支持点号分隔（如 "app.title"）
            default: 默认文本，如果找不到翻译则返回此值
            **kwargs: 文本格式化参数
        
        Returns:
            翻译后的文本
        
        示例:
            i18n.t("app.title")
            i18n.t("backtest.metrics_total_return")
            i18n.t("common.loading")
        """
        # 获取当前语言的翻译
        translation = self._translations.get(self.current_lang, {})
        
        # 按点号分割键
        keys = key.split(".")
        
        # 逐级查找
        value = translation
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # 找不到，尝试默认语言
                if self.current_lang != self.default_lang:
                    default_translation = self._translations.get(self.default_lang, {})
                    dv = default_translation
                    for dk in keys:
                        if isinstance(dv, dict) and dk in dv:
                            dv = dv[dk]
                        else:
                            dv = None
                            break
                    if dv is not None:
                        value = dv
                        break
                
                # 返回默认值或键名
                return default if default is not None else key
        
        # 格式化文本
        if isinstance(value, str) and kwargs:
            try:
                value = value.format(**kwargs)
            except KeyError:
                pass
        
        return value if isinstance(value, str) else (default if default is not None else key)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """获取支持的语言列表"""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def reload(self):
        """重新加载所有语言文件（用于热更新）"""
        self._translations.clear()
        self._load_all_translations()


# 全局实例（单例模式）
_i18n_instance: Optional[I18nManager] = None


def get_i18n() -> I18nManager:
    """获取全局 I18nManager 实例"""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = I18nManager()
    return _i18n_instance


def t(key: str, default: Optional[str] = None, **kwargs) -> str:
    """
    快捷翻译函数
    
    示例:
        from utils.i18n import t
        text = t("app.title")
    """
    return get_i18n().t(key, default, **kwargs)


def set_language(lang_code: str):
    """
    设置全局语言
    
    示例:
        from utils.i18n import set_language
        set_language("ja")
    """
    get_i18n().set_language(lang_code)


def get_current_language() -> str:
    """获取当前语言代码"""
    return get_i18n().get_language()
