#!/usr/bin/env python3
"""
Keyboard Shortcut Customization System
Fully customizable keyboard shortcuts with conflict detection and profiles
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import platform
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModifierKey(Enum):
    """Keyboard modifier keys"""
    CTRL = "ctrl"
    CMD = "cmd"      # macOS Command key
    ALT = "alt"
    SHIFT = "shift"
    META = "meta"    # Windows key
    FN = "fn"        # Function key

class ShortcutContext(Enum):
    """Context where shortcut is active"""
    GLOBAL = "global"
    TIMELINE = "timeline"
    VIEWER = "viewer"
    MEDIA_POOL = "media_pool"
    INSPECTOR = "inspector"
    EFFECTS = "effects"
    COLOR = "color"
    AUDIO = "audio"
    TEXT_EDITOR = "text_editor"

class ShortcutCategory(Enum):
    """Shortcut categories for organization"""
    FILE = "file"
    EDIT = "edit"
    VIEW = "view"
    PLAYBACK = "playback"
    TIMELINE = "timeline"
    TOOLS = "tools"
    EFFECTS = "effects"
    WINDOWS = "windows"
    HELP = "help"
    CUSTOM = "custom"

@dataclass
class KeyCombination:
    """Represents a key combination"""
    key: str
    modifiers: List[ModifierKey]
    
    def to_string(self) -> str:
        """Convert to human-readable string"""
        mod_str = "+".join([m.value.capitalize() for m in self.modifiers])
        if mod_str:
            return f"{mod_str}+{self.key.upper()}"
        return self.key.upper()
    
    def to_platform_string(self) -> str:
        """Convert to platform-specific string"""
        system = platform.system()
        
        # Map modifiers to platform-specific keys
        platform_mods = []
        for mod in self.modifiers:
            if mod == ModifierKey.CMD:
                if system == "Darwin":
                    platform_mods.append("⌘")
                else:
                    platform_mods.append("Ctrl")
            elif mod == ModifierKey.CTRL:
                if system == "Darwin":
                    platform_mods.append("⌃")
                else:
                    platform_mods.append("Ctrl")
            elif mod == ModifierKey.ALT:
                if system == "Darwin":
                    platform_mods.append("⌥")
                else:
                    platform_mods.append("Alt")
            elif mod == ModifierKey.SHIFT:
                if system == "Darwin":
                    platform_mods.append("⇧")
                else:
                    platform_mods.append("Shift")
            else:
                platform_mods.append(mod.value.capitalize())
        
        mod_str = "".join(platform_mods) if system == "Darwin" else "+".join(platform_mods)
        
        if mod_str:
            return f"{mod_str}{self.key.upper()}"
        return self.key.upper()
    
    def matches(self, key: str, modifiers: Set[str]) -> bool:
        """Check if this combination matches input"""
        if key.lower() != self.key.lower():
            return False
        
        expected_mods = set(m.value for m in self.modifiers)
        return expected_mods == modifiers

@dataclass
class Shortcut:
    """Single keyboard shortcut"""
    id: str
    name: str
    description: str
    category: ShortcutCategory
    context: ShortcutContext
    key_combination: KeyCombination
    action: str  # Action identifier
    enabled: bool = True
    custom: bool = False
    default_combination: Optional[KeyCombination] = None

@dataclass
class ShortcutProfile:
    """Collection of shortcuts"""
    name: str
    description: str
    shortcuts: Dict[str, Shortcut]
    is_default: bool = False
    parent_profile: Optional[str] = None  # For inheritance

class ShortcutConflict:
    """Represents a shortcut conflict"""
    
    def __init__(
        self,
        combination: KeyCombination,
        context: ShortcutContext,
        shortcuts: List[Shortcut]
    ):
        self.combination = combination
        self.context = context
        self.shortcuts = shortcuts
    
    def resolve(self, winner_id: str) -> List[Shortcut]:
        """Resolve conflict by choosing winner"""
        losers = []
        for shortcut in self.shortcuts:
            if shortcut.id != winner_id:
                losers.append(shortcut)
        return losers

class ShortcutManager:
    """Manages keyboard shortcuts"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.profiles: Dict[str, ShortcutProfile] = {}
        self.active_profile: Optional[str] = None
        self.action_handlers: Dict[str, Callable] = {}
        self.shortcut_index: Dict[Tuple[str, ShortcutContext], List[Shortcut]] = defaultdict(list)
        
        # Load default shortcuts
        self._load_defaults()
        
        # Load user profiles
        self._load_profiles()
    
    def _load_defaults(self):
        """Load default shortcuts"""
        defaults = self._get_default_shortcuts()
        
        profile = ShortcutProfile(
            name="Default",
            description="Default AutoResolve shortcuts",
            shortcuts=defaults,
            is_default=True
        )
        
        self.profiles["default"] = profile
        self.active_profile = "default"
        self._rebuild_index()
    
    def _get_default_shortcuts(self) -> Dict[str, Shortcut]:
        """Get default shortcut definitions"""
        system = platform.system()
        cmd_key = ModifierKey.CMD if system == "Darwin" else ModifierKey.CTRL
        
        defaults = {
            # File operations
            "file.new": Shortcut(
                id="file.new",
                name="New Project",
                description="Create new project",
                category=ShortcutCategory.FILE,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("n", [cmd_key]),
                action="new_project"
            ),
            "file.open": Shortcut(
                id="file.open",
                name="Open Project",
                description="Open existing project",
                category=ShortcutCategory.FILE,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("o", [cmd_key]),
                action="open_project"
            ),
            "file.save": Shortcut(
                id="file.save",
                name="Save Project",
                description="Save current project",
                category=ShortcutCategory.FILE,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("s", [cmd_key]),
                action="save_project"
            ),
            
            # Edit operations
            "edit.undo": Shortcut(
                id="edit.undo",
                name="Undo",
                description="Undo last action",
                category=ShortcutCategory.EDIT,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("z", [cmd_key]),
                action="undo"
            ),
            "edit.redo": Shortcut(
                id="edit.redo",
                name="Redo",
                description="Redo last undone action",
                category=ShortcutCategory.EDIT,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("z", [cmd_key, ModifierKey.SHIFT]),
                action="redo"
            ),
            "edit.cut": Shortcut(
                id="edit.cut",
                name="Cut",
                description="Cut selection",
                category=ShortcutCategory.EDIT,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("x", [cmd_key]),
                action="cut"
            ),
            "edit.copy": Shortcut(
                id="edit.copy",
                name="Copy",
                description="Copy selection",
                category=ShortcutCategory.EDIT,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("c", [cmd_key]),
                action="copy"
            ),
            "edit.paste": Shortcut(
                id="edit.paste",
                name="Paste",
                description="Paste from clipboard",
                category=ShortcutCategory.EDIT,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("v", [cmd_key]),
                action="paste"
            ),
            
            # Playback controls
            "playback.play": Shortcut(
                id="playback.play",
                name="Play/Pause",
                description="Toggle playback",
                category=ShortcutCategory.PLAYBACK,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("space", []),
                action="toggle_playback"
            ),
            "playback.stop": Shortcut(
                id="playback.stop",
                name="Stop",
                description="Stop playback",
                category=ShortcutCategory.PLAYBACK,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("k", []),
                action="stop_playback"
            ),
            "playback.forward": Shortcut(
                id="playback.forward",
                name="Step Forward",
                description="Step forward one frame",
                category=ShortcutCategory.PLAYBACK,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("right", []),
                action="step_forward"
            ),
            "playback.backward": Shortcut(
                id="playback.backward",
                name="Step Backward",
                description="Step backward one frame",
                category=ShortcutCategory.PLAYBACK,
                context=ShortcutContext.GLOBAL,
                key_combination=KeyCombination("left", []),
                action="step_backward"
            ),
            
            # Timeline operations
            "timeline.blade": Shortcut(
                id="timeline.blade",
                name="Blade Tool",
                description="Split clip at playhead",
                category=ShortcutCategory.TIMELINE,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("b", []),
                action="blade_tool"
            ),
            "timeline.select": Shortcut(
                id="timeline.select",
                name="Selection Tool",
                description="Activate selection tool",
                category=ShortcutCategory.TIMELINE,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("a", []),
                action="selection_tool"
            ),
            "timeline.ripple": Shortcut(
                id="timeline.ripple",
                name="Ripple Delete",
                description="Delete and close gap",
                category=ShortcutCategory.TIMELINE,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("delete", [ModifierKey.SHIFT]),
                action="ripple_delete"
            ),
            
            # View operations
            "view.zoom_in": Shortcut(
                id="view.zoom_in",
                name="Zoom In",
                description="Zoom in timeline",
                category=ShortcutCategory.VIEW,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("=", [cmd_key]),
                action="zoom_in"
            ),
            "view.zoom_out": Shortcut(
                id="view.zoom_out",
                name="Zoom Out",
                description="Zoom out timeline",
                category=ShortcutCategory.VIEW,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("-", [cmd_key]),
                action="zoom_out"
            ),
            "view.fit": Shortcut(
                id="view.fit",
                name="Fit to Window",
                description="Fit timeline to window",
                category=ShortcutCategory.VIEW,
                context=ShortcutContext.TIMELINE,
                key_combination=KeyCombination("0", [cmd_key]),
                action="fit_to_window"
            ),
        }
        
        return defaults
    
    def _load_profiles(self):
        """Load user-defined profiles"""
        profiles_dir = self.config_dir / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        for profile_file in profiles_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct shortcuts
                shortcuts = {}
                for shortcut_id, shortcut_data in data.get("shortcuts", {}).items():
                    key_combo = KeyCombination(
                        key=shortcut_data["key"],
                        modifiers=[ModifierKey(m) for m in shortcut_data["modifiers"]]
                    )
                    
                    shortcut = Shortcut(
                        id=shortcut_id,
                        name=shortcut_data["name"],
                        description=shortcut_data["description"],
                        category=ShortcutCategory(shortcut_data["category"]),
                        context=ShortcutContext(shortcut_data["context"]),
                        key_combination=key_combo,
                        action=shortcut_data["action"],
                        enabled=shortcut_data.get("enabled", True),
                        custom=shortcut_data.get("custom", False)
                    )
                    
                    shortcuts[shortcut_id] = shortcut
                
                profile = ShortcutProfile(
                    name=data["name"],
                    description=data["description"],
                    shortcuts=shortcuts,
                    is_default=False,
                    parent_profile=data.get("parent_profile")
                )
                
                self.profiles[profile_file.stem] = profile
                logger.info(f"Loaded profile: {profile.name}")
                
            except Exception as e:
                logger.error(f"Failed to load profile {profile_file}: {e}")
    
    def save_profile(self, profile_id: str):
        """Save profile to disk"""
        if profile_id not in self.profiles:
            return
        
        profile = self.profiles[profile_id]
        
        if profile.is_default:
            logger.warning("Cannot save default profile")
            return
        
        # Convert to serializable format
        data = {
            "name": profile.name,
            "description": profile.description,
            "parent_profile": profile.parent_profile,
            "shortcuts": {}
        }
        
        for shortcut_id, shortcut in profile.shortcuts.items():
            data["shortcuts"][shortcut_id] = {
                "name": shortcut.name,
                "description": shortcut.description,
                "category": shortcut.category.value,
                "context": shortcut.context.value,
                "key": shortcut.key_combination.key,
                "modifiers": [m.value for m in shortcut.key_combination.modifiers],
                "action": shortcut.action,
                "enabled": shortcut.enabled,
                "custom": shortcut.custom
            }
        
        # Save to file
        profiles_dir = self.config_dir / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        
        profile_file = profiles_dir / f"{profile_id}.json"
        with open(profile_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved profile: {profile.name}")
    
    def create_profile(self, name: str, description: str, base_profile: str = "default") -> str:
        """Create new profile based on existing one"""
        profile_id = name.lower().replace(" ", "_")
        
        if profile_id in self.profiles:
            logger.warning(f"Profile {profile_id} already exists")
            return profile_id
        
        # Copy shortcuts from base profile
        base = self.profiles.get(base_profile, self.profiles["default"])
        shortcuts = {}
        
        for shortcut_id, shortcut in base.shortcuts.items():
            shortcuts[shortcut_id] = Shortcut(
                id=shortcut.id,
                name=shortcut.name,
                description=shortcut.description,
                category=shortcut.category,
                context=shortcut.context,
                key_combination=shortcut.key_combination,
                action=shortcut.action,
                enabled=shortcut.enabled,
                custom=shortcut.custom,
                default_combination=shortcut.key_combination
            )
        
        profile = ShortcutProfile(
            name=name,
            description=description,
            shortcuts=shortcuts,
            is_default=False,
            parent_profile=base_profile
        )
        
        self.profiles[profile_id] = profile
        self.save_profile(profile_id)
        
        return profile_id
    
    def set_active_profile(self, profile_id: str):
        """Set active shortcut profile"""
        if profile_id not in self.profiles:
            logger.error(f"Profile not found: {profile_id}")
            return
        
        self.active_profile = profile_id
        self._rebuild_index()
        logger.info(f"Activated profile: {profile_id}")
    
    def customize_shortcut(
        self,
        shortcut_id: str,
        new_combination: KeyCombination,
        profile_id: Optional[str] = None
    ) -> List[ShortcutConflict]:
        """Customize a shortcut"""
        if not profile_id:
            profile_id = self.active_profile
        
        if profile_id not in self.profiles:
            logger.error(f"Profile not found: {profile_id}")
            return []
        
        profile = self.profiles[profile_id]
        
        if shortcut_id not in profile.shortcuts:
            logger.error(f"Shortcut not found: {shortcut_id}")
            return []
        
        # Check for conflicts
        conflicts = self.check_conflicts(new_combination, profile_id)
        
        # Update shortcut
        shortcut = profile.shortcuts[shortcut_id]
        shortcut.key_combination = new_combination
        shortcut.custom = True
        
        # Save profile
        if not profile.is_default:
            self.save_profile(profile_id)
        
        # Rebuild index
        self._rebuild_index()
        
        return conflicts
    
    def check_conflicts(
        self,
        combination: KeyCombination,
        profile_id: Optional[str] = None
    ) -> List[ShortcutConflict]:
        """Check for shortcut conflicts"""
        if not profile_id:
            profile_id = self.active_profile
        
        if profile_id not in self.profiles:
            return []
        
        profile = self.profiles[profile_id]
        conflicts_by_context: Dict[ShortcutContext, List[Shortcut]] = defaultdict(list)
        
        # Check all shortcuts in profile
        for shortcut in profile.shortcuts.values():
            if not shortcut.enabled:
                continue
            
            if shortcut.key_combination.to_string() == combination.to_string():
                conflicts_by_context[shortcut.context].append(shortcut)
        
        # Create conflict objects
        conflicts = []
        for context, shortcuts in conflicts_by_context.items():
            if len(shortcuts) > 1:
                conflicts.append(ShortcutConflict(combination, context, shortcuts))
        
        return conflicts
    
    def register_action(self, action: str, handler: Callable):
        """Register action handler"""
        self.action_handlers[action] = handler
        logger.debug(f"Registered action: {action}")
    
    def handle_keypress(
        self,
        key: str,
        modifiers: Set[str],
        context: ShortcutContext = ShortcutContext.GLOBAL
    ) -> bool:
        """Handle keyboard input"""
        if not self.active_profile:
            return False
        
        profile = self.profiles[self.active_profile]
        
        # Find matching shortcut
        for shortcut in profile.shortcuts.values():
            if not shortcut.enabled:
                continue
            
            # Check context (GLOBAL matches all contexts)
            if shortcut.context != ShortcutContext.GLOBAL and shortcut.context != context:
                continue
            
            # Check key combination
            if shortcut.key_combination.matches(key, modifiers):
                # Execute action
                if shortcut.action in self.action_handlers:
                    try:
                        self.action_handlers[shortcut.action]()
                        logger.debug(f"Executed action: {shortcut.action}")
                        return True
                    except Exception as e:
                        logger.error(f"Action failed: {e}")
                else:
                    logger.warning(f"No handler for action: {shortcut.action}")
        
        return False
    
    def _rebuild_index(self):
        """Rebuild shortcut index for fast lookup"""
        self.shortcut_index.clear()
        
        if not self.active_profile:
            return
        
        profile = self.profiles[self.active_profile]
        
        for shortcut in profile.shortcuts.values():
            if shortcut.enabled:
                key = (shortcut.key_combination.to_string(), shortcut.context)
                self.shortcut_index[key].append(shortcut)
    
    def export_profile(self, profile_id: str, output_path: str):
        """Export profile to file"""
        if profile_id not in self.profiles:
            logger.error(f"Profile not found: {profile_id}")
            return
        
        profile = self.profiles[profile_id]
        
        # Generate human-readable format
        output = []
        output.append(f"# {profile.name}")
        output.append(f"# {profile.description}")
        output.append("")
        
        # Group by category
        by_category: Dict[ShortcutCategory, List[Shortcut]] = defaultdict(list)
        for shortcut in profile.shortcuts.values():
            by_category[shortcut.category].append(shortcut)
        
        for category in ShortcutCategory:
            if category not in by_category:
                continue
            
            output.append(f"## {category.value.capitalize()}")
            output.append("")
            
            for shortcut in sorted(by_category[category], key=lambda s: s.name):
                combo_str = shortcut.key_combination.to_platform_string()
                output.append(f"{shortcut.name:<30} {combo_str:<20} {shortcut.description}")
            
            output.append("")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write("\n".join(output))
        
        logger.info(f"Exported profile to: {output_path}")
    
    def get_shortcuts_by_category(
        self,
        category: ShortcutCategory,
        profile_id: Optional[str] = None
    ) -> List[Shortcut]:
        """Get all shortcuts in a category"""
        if not profile_id:
            profile_id = self.active_profile
        
        if profile_id not in self.profiles:
            return []
        
        profile = self.profiles[profile_id]
        return [s for s in profile.shortcuts.values() if s.category == category]
    
    def reset_to_defaults(self, profile_id: Optional[str] = None):
        """Reset profile to default shortcuts"""
        if not profile_id:
            profile_id = self.active_profile
        
        if profile_id not in self.profiles:
            return
        
        profile = self.profiles[profile_id]
        
        # Reset each shortcut to default
        for shortcut in profile.shortcuts.values():
            if shortcut.default_combination:
                shortcut.key_combination = shortcut.default_combination
                shortcut.custom = False
        
        # Save and rebuild
        if not profile.is_default:
            self.save_profile(profile_id)
        
        self._rebuild_index()
        logger.info(f"Reset profile to defaults: {profile_id}")

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = ShortcutManager("/tmp/shortcuts")
    
    # Create custom profile
    profile_id = manager.create_profile(
        "My Custom Profile",
        "Personal shortcut configuration"
    )
    
    # Customize a shortcut
    new_combo = KeyCombination("p", [ModifierKey.CTRL, ModifierKey.SHIFT])
    conflicts = manager.customize_shortcut("playback.play", new_combo, profile_id)
    
    if conflicts:
        print(f"⚠️ Conflicts detected: {len(conflicts)}")
    
    # Register action handlers
    manager.register_action("toggle_playback", lambda: print("Play/Pause"))
    manager.register_action("save_project", lambda: print("Saving project..."))
    
    # Test keypress handling
    handled = manager.handle_keypress("space", set(), ShortcutContext.GLOBAL)
    print(f"✅ Keypress handled: {handled}")
    
    # Export profile
    manager.export_profile(profile_id, "/tmp/shortcuts.txt")
    print("✅ Keyboard shortcuts system ready")