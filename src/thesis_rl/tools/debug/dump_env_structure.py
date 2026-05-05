#!/usr/bin/env python
"""
Dump MetaDrive environment structure and attributes to JSON for inspection.
Run: uv run --no-sync python src/thesis_rl/tools/debug/dump_env_structure.py
"""
import json
import sys
from pathlib import Path
from typing import Any

def safe_serialize(obj: Any, depth: int = 0, max_depth: int = 5) -> Any:
    """Safely serialize an object to JSON-compatible format, with depth limit."""
    if depth > max_depth:
        return f"<MAX_DEPTH_EXCEEDED>"
    
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item, depth + 1, max_depth) for item in obj[:5]]  # limit list size
    
    if isinstance(obj, dict):
        # Convert all keys to strings (JSON requires string keys)
        result = {}
        for k, v in list(obj.items())[:10]:
            key_str = str(k) if not isinstance(k, str) else k
            result[key_str] = safe_serialize(v, depth + 1, max_depth)
        return result
    
    # For objects, try to extract useful info
    if hasattr(obj, '__dict__'):
        attrs = {}
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            try:
                val = getattr(obj, attr_name)
                if not callable(val):
                    attrs[attr_name] = safe_serialize(val, depth + 1, max_depth)
            except:
                pass
        if attrs:
            return {"<type>": type(obj).__name__, "<attrs>": attrs}
    
    return f"<{type(obj).__name__}>"

def main():
    from omegaconf import OmegaConf
    from thesis_rl.envs.factory import make_env
    
    # Load config for env
    cfg_yaml = """
name: metadrive
env_id: MetaDriveEnv
config:
  use_render: false
  horizon: 500
  map: 5
  traffic_density: 0.1
  start_seed: 10000
  num_scenarios: 5000
  random_spawn_lane_index: true
  random_lane_width: true
  random_lane_num: true
  random_agent_model: true
  physics_world_step_size: 0.02
  decision_repeat: 5
  log_level: 50
vectorized:
  enabled: true
  num_envs: 1
  start_method: forkserver
policy_mode:
  enabled: false
"""
    cfg = OmegaConf.create(cfg_yaml)
    
    print("=" * 80)
    print("Creating MetaDrive Environment...")
    print("=" * 80)
    
    env = make_env(cfg)
    print("✓ Environment created")
    
    print("\n" + "=" * 80)
    print("Resetting environment...")
    print("=" * 80)
    obs, info = env.reset()
    print(f"✓ Reset complete. obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    print("\n" + "=" * 80)
    print("Taking one step...")
    print("=" * 80)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"✓ Step complete. reward={reward}, terminated={terminated}, truncated={truncated}")
    
    print("\n" + "=" * 80)
    print("Extracting environment structure...")
    print("=" * 80)
    
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    dump = {
        "base_env_type": type(base_env).__name__,
        "base_env_module": type(base_env).__module__,
        
        # Top-level attributes
        "agents": safe_serialize(getattr(base_env, "agents", None)),
        "vehicle": safe_serialize(getattr(base_env, "vehicle", None)),
        "engine": safe_serialize(getattr(base_env, "engine", None)),
        "traffic_manager": safe_serialize(getattr(base_env, "traffic_manager", None)),
        "current_map": safe_serialize(getattr(base_env, "current_map", None)),
        "config": safe_serialize(getattr(base_env, "config", None)),
        
        # Info dict from last step
        "info_keys": list(info.keys()) if isinstance(info, dict) else "NOT_A_DICT",
        "info_sample": {k: safe_serialize(v) for k, v in list(info.items())[:5]} if isinstance(info, dict) else {},
        
        # Check for traffic info
        "step_info_keys": list(info.get("step_info", {}).keys()) if isinstance(info.get("step_info"), dict) else "N/A",
    }
    
    # Deep dive: engine attributes
    engine = getattr(base_env, "engine", None)
    if engine is not None:
        dump["engine_details"] = {
            "type": type(engine).__name__,
            "traffic_manager_via_engine": safe_serialize(getattr(engine, "traffic_manager", None)),
            "agents_via_engine": safe_serialize(getattr(engine, "agents", None)),
            "dynamic_objects": safe_serialize(getattr(engine, "dynamic_objects", None)),
        }
    
    # Deep dive: traffic manager
    traffic_manager = getattr(base_env, "traffic_manager", None)
    if traffic_manager is None and engine is not None:
        traffic_manager = getattr(engine, "traffic_manager", None)
    
    if traffic_manager is not None:
        dump["traffic_manager_details"] = {
            "type": type(traffic_manager).__name__,
            "attributes": [a for a in dir(traffic_manager) if not a.startswith('_')],
            "traffic_vehicles": safe_serialize(getattr(traffic_manager, "traffic_vehicles", None)),
            "vehicles": safe_serialize(getattr(traffic_manager, "vehicles", None)),
            "_traffic_vehicles": safe_serialize(getattr(traffic_manager, "_traffic_vehicles", None)),
        }
    
    # Output path
    output_file = Path("outputs/env_structure_dump.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    
    print(f"\n✓ Dump saved to: {output_file}")
    print("\n" + "=" * 80)
    print("Quick summary:")
    print("=" * 80)
    print(f"- Has agents: {dump['agents'] is not None and len(dump['agents']) > 0 if isinstance(dump['agents'], dict) else 'N/A'}")
    print(f"- Has vehicle (ego): {dump['vehicle'] is not None}")
    print(f"- Has traffic_manager: {dump['traffic_manager'] is not None}")
    print(f"- Traffic manager details available: {'traffic_manager_details' in dump}")
    print(f"- Info dict keys: {dump['info_keys']}")

if __name__ == "__main__":
    main()
