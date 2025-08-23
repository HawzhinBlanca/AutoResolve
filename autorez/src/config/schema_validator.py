"""
Configuration schema validation for AutoResolve.
Ensures all config values are within valid ranges and types.
"""

from typing import Dict, Any
import configparser
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration files against defined schemas."""
    
    # Define validation schemas for each config section
    SCHEMAS = {
        'embeddings': {
            'default_model': {
                'type': str, 
                'choices': ['clip', 'vjepa'],
                'required': True
            },
            'backend': {
                'type': str, 
                'choices': ['mps', 'cuda', 'cpu'],
                'required': True
            },
            'fps': {
                'type': float, 
                'min': 0.1, 
                'max': 30.0,
                'default': 1.0
            },
            'window': {
                'type': int, 
                'min': 1, 
                'max': 64,
                'default': 16
            },
            'crop': {
                'type': int, 
                'min': 112, 
                'max': 512,
                'choices': [112, 224, 256, 384, 512],  # Common sizes
                'default': 256
            },
            'max_rss_gb': {
                'type': float, 
                'min': 4, 
                'max': 128,
                'default': 16
            },
            'max_segments': {
                'type': int, 
                'min': 1, 
                'max': 10000,
                'default': 500
            },
            'seed': {
                'type': int, 
                'min': 0,
                'default': 1234
            },
            'strategy': {
                'type': str,
                'choices': ['cls', 'patch_mean', 'temp_attn'],
                'default': 'temp_attn'
            }
        },
        'director.narrative': {
            'stable_momentum': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.10
            },
            'spike_momentum': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.80
            },
            'novelty_thresh': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.30
            },
            'high_energy_pct': {
                'type': int,
                'min': 50,
                'max': 95,
                'default': 75
            }
        },
        'ops.silence': {
            'rms_thresh_db': {
                'type': float, 
                'max': 0,  # Must be negative
                'min': -60,
                'default': -34
            },
            'min_silence_s': {
                'type': float, 
                'min': 0.1, 
                'max': 5.0,
                'default': 0.35
            },
            'min_keep_s': {
                'type': float, 
                'min': 0.1, 
                'max': 10.0,
                'default': 0.40
            },
            'pad_s': {
                'type': float,
                'min': 0.0,
                'max': 1.0,
                'default': 0.05
            }
        },
        'ops.shorts': {
            'target': {
                'type': int,
                'min': 15,
                'max': 180,
                'default': 60
            },
            'min_seg': {
                'type': float,
                'min': 1.0,
                'max': 30.0,
                'default': 3.0
            },
            'max_seg': {
                'type': float,
                'min': 3.0,
                'max': 60.0,
                'default': 18.0
            },
            'topk': {
                'type': int,
                'min': 1,
                'max': 100,
                'default': 12
            }
        },
        'broll': {
            'max_overlay_s': {
                'type': float,
                'min': 1.0,
                'max': 30.0,
                'default': 7.0
            },
            'min_gap_s': {
                'type': float,
                'min': 1.0,
                'max': 60.0,
                'default': 4.0
            },
            'dissolve_s': {
                'type': float,
                'min': 0.1,
                'max': 2.0,
                'default': 0.25
            }
        }
    }
    
    @classmethod
    def validate_config(cls, config_path: str, section: str = None) -> Dict[str, Any]:
        """
        Validate configuration file against schema.
        
        Args:
            config_path: Path to configuration file
            section: Specific section to validate (None for all)
            
        Returns:
            Dict of validated values with defaults applied
            
        Raises:
            ValueError: If validation fails
        """
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        results = {}
        sections_to_validate = [section] if section else cls.SCHEMAS.keys()
        
        for sec in sections_to_validate:
            if sec not in cls.SCHEMAS:
                continue
                
            schema = cls.SCHEMAS[sec]
            validated_section = {}
            
            # Handle nested sections (e.g., "ops.silence")
            config_section = sec.split('.')[-1] if '.' in sec else sec
            
            for key, constraints in schema.items():
                # Get value from config
                if parser.has_section(config_section):
                    value_str = parser.get(config_section, key, fallback=None)
                else:
                    value_str = None
                
                # Apply default if not specified
                if value_str is None:
                    if 'default' in constraints:
                        value = constraints['default']
                    elif constraints.get('required', False):
                        raise ValueError(f"Required config {sec}.{key} not found")
                    else:
                        continue
                else:
                    # Type conversion
                    value = cls._convert_type(value_str, constraints['type'])
                
                # Validate value
                cls._validate_value(key, value, constraints)
                validated_section[key] = value
            
            results[sec] = validated_section
        
        return results
    
    @staticmethod
    def _convert_type(value_str: str, target_type: type) -> Any:
        """Convert string value to target type."""
        if target_type == bool:
            return value_str.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(float(value_str))  # Handle int values passed as floats
        elif target_type == float:
            return float(value_str)
        else:
            return value_str
    
    @staticmethod
    def _validate_value(key: str, value: Any, constraints: Dict) -> None:
        """Validate a single value against constraints."""
        # Check type (allow int for float fields)
        expected_type = constraints['type']
        if expected_type == float and isinstance(value, int):
            # Allow integers for float fields
            pass
        elif not isinstance(value, expected_type):
            raise ValueError(f"{key}: expected {expected_type.__name__}, got {type(value).__name__}")
        
        # Check choices
        if 'choices' in constraints and value not in constraints['choices']:
            raise ValueError(f"{key} = {value} not in allowed choices: {constraints['choices']}")
        
        # Check numeric bounds
        if 'min' in constraints and value < constraints['min']:
            raise ValueError(f"{key} = {value} below minimum {constraints['min']}")
        
        if 'max' in constraints and value > constraints['max']:
            raise ValueError(f"{key} = {value} above maximum {constraints['max']}")
    
    @classmethod
    def generate_default_config(cls, section: str) -> str:
        """Generate default configuration for a section."""
        if section not in cls.SCHEMAS:
            raise ValueError(f"Unknown section: {section}")
        
        lines = [f"[{section.split('.')[-1]}]"]
        for key, constraints in cls.SCHEMAS[section].items():
            if 'default' in constraints:
                value = constraints['default']
                comment = ""
                if 'choices' in constraints:
                    comment = f"  ; choices: {constraints['choices']}"
                elif 'min' in constraints or 'max' in constraints:
                    bounds = []
                    if 'min' in constraints:
                        bounds.append(f"min={constraints['min']}")
                    if 'max' in constraints:
                        bounds.append(f"max={constraints['max']}")
                    comment = f"  ; {', '.join(bounds)}"
                lines.append(f"{key} = {value}{comment}")
        
        return "\n".join(lines)