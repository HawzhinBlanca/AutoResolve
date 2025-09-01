#!/usr/bin/env python3
"""
Plugin System Architecture
Extensible plugin framework for AutoResolve with hot-reloading and sandboxing
"""

import os
import sys
import json
import yaml
import importlib
import importlib.util
import inspect
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import threading
from datetime import datetime
import ast
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Plugin categories"""
    EFFECT = "effect"           # Video/audio effects
    TRANSITION = "transition"    # Transitions between clips
    GENERATOR = "generator"      # Content generators
    ANALYZER = "analyzer"        # Analysis tools
    EXPORTER = "exporter"       # Export formats
    IMPORTER = "importer"       # Import formats
    FILTER = "filter"           # Processing filters
    TOOL = "tool"               # Utility tools
    AI_MODEL = "ai_model"       # AI/ML models
    UI_COMPONENT = "ui_component"  # UI extensions

class PluginStatus(Enum):
    """Plugin status"""
    LOADED = "loaded"
    UNLOADED = "unloaded"
    ERROR = "error"
    DISABLED = "disabled"
    UPDATING = "updating"

class PluginPermission(Enum):
    """Plugin permissions"""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK = "network"
    SYSTEM_EXEC = "system_exec"
    GPU_ACCESS = "gpu_access"
    MEMORY_UNLIMITED = "memory_unlimited"
    UI_MODIFY = "ui_modify"
    TIMELINE_EDIT = "timeline_edit"

@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    api_version: str = "1.0"
    homepage: Optional[str] = None
    license: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    permissions: List[PluginPermission] = field(default_factory=list)
    settings_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class PluginContext:
    """Runtime context for plugins"""
    project_path: str
    timeline_id: str
    selection: List[str]
    current_time: float
    fps: float
    resolution: Tuple[int, int]
    user_settings: Dict[str, Any]
    temp_dir: str

class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self, context: PluginContext) -> bool:
        """Initialize plugin"""
        pass
    
    @abstractmethod
    def execute(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Execute plugin functionality"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters"""
        return True
    
    def get_preview(self, input_data: Any, parameters: Dict[str, Any]) -> Optional[Any]:
        """Generate preview of plugin effect"""
        return None

class EffectPlugin(PluginInterface):
    """Base class for effect plugins"""
    
    @abstractmethod
    def apply_effect(self, frame: Any, parameters: Dict[str, Any]) -> Any:
        """Apply effect to frame"""
        pass
    
    def execute(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Execute effect"""
        return self.apply_effect(input_data, parameters)

class TransitionPlugin(PluginInterface):
    """Base class for transition plugins"""
    
    @abstractmethod
    def create_transition(
        self,
        frame_a: Any,
        frame_b: Any,
        progress: float,
        parameters: Dict[str, Any]
    ) -> Any:
        """Create transition between frames"""
        pass
    
    def execute(self, input_data: Any, parameters: Dict[str, Any]) -> Any:
        """Execute transition"""
        frame_a = input_data.get("frame_a")
        frame_b = input_data.get("frame_b")
        progress = parameters.get("progress", 0.5)
        return self.create_transition(frame_a, frame_b, progress, parameters)

class PluginSandbox:
    """Sandbox environment for plugin execution"""
    
    def __init__(self, permissions: List[PluginPermission]):
        self.permissions = permissions
        self.restricted_modules = [
            'os', 'sys', 'subprocess', 'socket', 'urllib',
            '__builtin__', '__builtins__', 'eval', 'exec'
        ]
        
    def create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace"""
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        }
        
        restricted_globals = {
            '__builtins__': safe_builtins,
            '__name__': 'plugin_sandbox',
            '__doc__': None,
        }
        
        # Add permitted modules based on permissions
        if PluginPermission.FILE_READ in self.permissions:
            restricted_globals['open'] = self._restricted_open_read
        
        if PluginPermission.FILE_WRITE in self.permissions:
            restricted_globals['open'] = open  # Full open access
        
        return restricted_globals
    
    def _restricted_open_read(self, file, mode='r', *args, **kwargs):
        """Restricted file open for reading only"""
        if 'w' in mode or 'a' in mode or '+' in mode:
            raise PermissionError("Write access not permitted")
        return open(file, mode, *args, **kwargs)
    
    def validate_code(self, code: str) -> bool:
        """Validate plugin code for security issues"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.restricted_modules:
                            logger.warning(f"Restricted module import: {alias.name}")
                            return False
                
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', '__import__']:
                            logger.warning(f"Dangerous function call: {node.func.id}")
                            return False
            
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in plugin code: {e}")
            return False

class PluginLoader:
    """Dynamic plugin loader"""
    
    def __init__(self, plugin_dir: str):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        
    def load_plugin(self, plugin_path: str) -> Optional[PluginInterface]:
        """Load a plugin from file"""
        plugin_path = Path(plugin_path)
        
        if not plugin_path.exists():
            logger.error(f"Plugin not found: {plugin_path}")
            return None
        
        try:
            # Load metadata
            metadata = self._load_metadata(plugin_path)
            if not metadata:
                logger.error(f"Invalid plugin metadata: {plugin_path}")
                return None
            
            # Create sandbox
            sandbox = PluginSandbox(metadata.permissions)
            
            # Validate code
            with open(plugin_path, 'r') as f:
                code = f.read()
            
            if not sandbox.validate_code(code):
                logger.error(f"Plugin failed security validation: {plugin_path}")
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_path.stem}",
                plugin_path
            )
            
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec: {plugin_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Set restricted globals
            module.__dict__.update(sandbox.create_restricted_globals())
            
            # Execute module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                logger.error(f"No plugin class found: {plugin_path}")
                return None
            
            # Instantiate plugin
            plugin_instance = plugin_class()
            
            # Store plugin
            plugin_id = f"{metadata.name}_{metadata.version}"
            self.loaded_plugins[plugin_id] = plugin_instance
            self.plugin_metadata[plugin_id] = metadata
            
            logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return None
    
    def _load_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from manifest"""
        manifest_path = plugin_path.parent / f"{plugin_path.stem}.yaml"
        
        if not manifest_path.exists():
            manifest_path = plugin_path.parent / f"{plugin_path.stem}.json"
        
        if not manifest_path.exists():
            # Try to extract from plugin file
            return self._extract_metadata_from_code(plugin_path)
        
        try:
            with open(manifest_path, 'r') as f:
                if manifest_path.suffix == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return PluginMetadata(
                name=data['name'],
                version=data['version'],
                author=data['author'],
                description=data['description'],
                plugin_type=PluginType(data['type']),
                api_version=data.get('api_version', '1.0'),
                homepage=data.get('homepage'),
                license=data.get('license'),
                dependencies=data.get('dependencies', []),
                permissions=[PluginPermission(p) for p in data.get('permissions', [])],
                settings_schema=data.get('settings_schema'),
                tags=data.get('tags', [])
            )
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def _extract_metadata_from_code(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Extract metadata from plugin code"""
        try:
            with open(plugin_path, 'r') as f:
                code = f.read()
            
            # Look for metadata in docstring or comments
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check class docstring
                    docstring = ast.get_docstring(node)
                    if docstring and 'PLUGIN_METADATA' in docstring:
                        # Parse metadata from docstring
                        metadata_str = docstring.split('PLUGIN_METADATA')[1].strip()
                        metadata_dict = json.loads(metadata_str)
                        
                        return PluginMetadata(**metadata_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return None
    
    def _find_plugin_class(self, module) -> Optional[Type[PluginInterface]]:
        """Find plugin class in module"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, PluginInterface):
                if obj != PluginInterface:  # Skip base class
                    return obj
        return None
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        if plugin_id in self.loaded_plugins:
            try:
                # Call cleanup
                plugin = self.loaded_plugins[plugin_id]
                plugin.cleanup()
                
                # Remove from registry
                del self.loaded_plugins[plugin_id]
                del self.plugin_metadata[plugin_id]
                
                logger.info(f"Unloaded plugin: {plugin_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {plugin_id}: {e}")
        
        return False
    
    def reload_plugin(self, plugin_id: str) -> bool:
        """Reload a plugin"""
        if plugin_id in self.plugin_metadata:
            metadata = self.plugin_metadata[plugin_id]
            
            # Unload
            self.unload_plugin(plugin_id)
            
            # Reload
            plugin_path = self.plugin_dir / f"{metadata.name}.py"
            if self.load_plugin(str(plugin_path)):
                return True
        
        return False

class PluginManager:
    """Central plugin management system"""
    
    def __init__(self, plugin_dir: str):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)
        self.loader = PluginLoader(plugin_dir)
        self.registry: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self.hot_reload_enabled = False
        self.file_watcher = None
        
    def discover_plugins(self):
        """Discover and load all plugins"""
        plugin_files = list(self.plugin_dir.glob("*.py"))
        
        for plugin_file in plugin_files:
            if plugin_file.stem.startswith("_"):
                continue  # Skip private files
            
            plugin = self.loader.load_plugin(str(plugin_file))
            if plugin:
                metadata = plugin.get_metadata()
                plugin_id = f"{metadata.name}_{metadata.version}"
                
                # Register by type
                if metadata.plugin_type not in self.registry:
                    self.registry[metadata.plugin_type] = []
                
                self.registry[metadata.plugin_type].append(plugin_id)
        
        logger.info(f"Discovered {len(self.loader.loaded_plugins)} plugins")
    
    def install_plugin(self, plugin_path: str) -> bool:
        """Install a plugin from external source"""
        source_path = Path(plugin_path)
        
        if not source_path.exists():
            logger.error(f"Plugin source not found: {plugin_path}")
            return False
        
        try:
            # Copy to plugin directory
            dest_path = self.plugin_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            # Copy manifest if exists
            manifest_extensions = ['.yaml', '.json']
            for ext in manifest_extensions:
                manifest_src = source_path.parent / f"{source_path.stem}{ext}"
                if manifest_src.exists():
                    manifest_dest = self.plugin_dir / manifest_src.name
                    shutil.copy2(manifest_src, manifest_dest)
                    break
            
            # Load plugin
            plugin = self.loader.load_plugin(str(dest_path))
            if plugin:
                metadata = plugin.get_metadata()
                plugin_id = f"{metadata.name}_{metadata.version}"
                
                # Register
                if metadata.plugin_type not in self.registry:
                    self.registry[metadata.plugin_type] = []
                
                self.registry[metadata.plugin_type].append(plugin_id)
                
                logger.info(f"Installed plugin: {metadata.name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to install plugin: {e}")
        
        return False
    
    def execute_plugin(
        self,
        plugin_id: str,
        context: PluginContext,
        input_data: Any,
        parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """Execute a plugin"""
        if plugin_id not in self.loader.loaded_plugins:
            logger.error(f"Plugin not found: {plugin_id}")
            return None
        
        plugin = self.loader.loaded_plugins[plugin_id]
        
        try:
            # Initialize if needed
            if not hasattr(plugin, '_initialized'):
                if plugin.initialize(context):
                    plugin._initialized = True
                else:
                    logger.error(f"Failed to initialize plugin: {plugin_id}")
                    return None
            
            # Validate parameters
            if not plugin.validate_parameters(parameters):
                logger.error(f"Invalid parameters for plugin: {plugin_id}")
                return None
            
            # Execute
            result = plugin.execute(input_data, parameters)
            
            return result
            
        except Exception as e:
            logger.error(f"Plugin execution failed: {e}")
            return None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """Get all plugins of a specific type"""
        return self.registry.get(plugin_type, [])
    
    def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata"""
        return self.loader.plugin_metadata.get(plugin_id)
    
    def enable_hot_reload(self):
        """Enable hot reloading of plugins"""
        if self.hot_reload_enabled:
            return
        
        self.hot_reload_enabled = True
        
        # Setup file watcher
        event_handler = PluginFileHandler(self)
        self.file_watcher = Observer()
        self.file_watcher.schedule(event_handler, str(self.plugin_dir), recursive=False)
        self.file_watcher.start()
        
        logger.info("Hot reload enabled for plugins")
    
    def disable_hot_reload(self):
        """Disable hot reloading"""
        if self.file_watcher:
            self.file_watcher.stop()
            self.file_watcher.join()
            self.file_watcher = None
        
        self.hot_reload_enabled = False
        logger.info("Hot reload disabled")

class PluginFileHandler(FileSystemEventHandler):
    """File system event handler for plugin hot reloading"""
    
    def __init__(self, manager: PluginManager):
        self.manager = manager
        
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.py'):
            plugin_path = Path(event.src_path)
            plugin_name = plugin_path.stem
            
            # Find plugin ID
            for plugin_id in self.manager.loader.loaded_plugins:
                if plugin_name in plugin_id:
                    logger.info(f"Reloading plugin: {plugin_id}")
                    self.manager.loader.reload_plugin(plugin_id)
                    break

# Example plugin implementation
class ExampleBlurEffect(EffectPlugin):
    """
    Example blur effect plugin
    
    PLUGIN_METADATA
    {
        "name": "blur_effect",
        "version": "1.0.0",
        "author": "AutoResolve",
        "description": "Gaussian blur effect",
        "type": "effect"
    }
    """
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="blur_effect",
            version="1.0.0",
            author="AutoResolve",
            description="Gaussian blur effect",
            plugin_type=PluginType.EFFECT,
            permissions=[],
            settings_schema={
                "radius": {"type": "float", "default": 5.0, "min": 0.0, "max": 50.0}
            }
        )
    
    def initialize(self, context: PluginContext) -> bool:
        return True
    
    def apply_effect(self, frame: Any, parameters: Dict[str, Any]) -> Any:
        # Apply Gaussian blur
        import cv2
        radius = parameters.get("radius", 5.0)
        kernel_size = int(radius * 2) | 1  # Ensure odd number
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), radius)
    
    def cleanup(self):
        pass

# Example usage
if __name__ == "__main__":
    # Initialize plugin manager
    manager = PluginManager("/tmp/plugins")
    
    # Discover plugins
    manager.discover_plugins()
    
    # Enable hot reload
    manager.enable_hot_reload()
    
    # Get effect plugins
    effects = manager.get_plugins_by_type(PluginType.EFFECT)
    print(f"✅ Found {len(effects)} effect plugins")
    
    # Execute a plugin
    if effects:
        context = PluginContext(
            project_path="/tmp/project",
            timeline_id="timeline-001",
            selection=[],
            current_time=0.0,
            fps=30.0,
            resolution=(1920, 1080),
            user_settings={},
            temp_dir="/tmp"
        )
        
        # Dummy frame data
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        result = manager.execute_plugin(
            effects[0],
            context,
            frame,
            {"radius": 10.0}
        )
        
        if result is not None:
            print(f"✅ Plugin executed successfully")