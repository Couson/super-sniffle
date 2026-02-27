# 🏗️ Text-to-3D Scene Generator

Generate 3D scenes from natural language using a **three-stage architecture with feedback loop**.

![Demo](assets/demo.gif)

## Architecture

```
User Prompt → [Stage 1: LLM] → Entities → [Stage 2: Rules] → Render → Image
                    ↑                                                   ↓
                    └─────────────── [Stage 3: Vision Critic] ←─────────┘
```

| Stage | Component | Role |
|-------|-----------|------|
| 1 | GPT-4o (LLM) | Scene understanding → high-level entities with parameters |
| 2 | EntityAgent (Rules) | **Deterministic** decomposition → geometric primitives (no LLM) |
| 3 | Vision Critic (LLM) | Analyze rendered image → score & feedback → refinement loop |

**Why three stages?**
- **Stage 1**: LLM focuses on *what* to create, not *how* to build it
- **Stage 2**: Entity decomposition is **deterministic** — no LLM calls, just rules
- **Stage 3**: Vision model validates output and triggers automatic refinement
- Self-improving pipeline that iterates until quality target is met

## Examples

### Medieval Village
```
"a medieval village with a church, market square, fountain, and houses"
```
![Medieval Village](assets/medieval_village.png)

### Mountain Lake
```
"a snowy mountain range with pine trees and a frozen lake at the base"
```
![Mountain Lake](assets/mountain_lake.png)

### Harbor Scene
```
"a coastal harbor with a lighthouse, wooden pier, boats, and a stone bridge"
```
![Harbor](assets/harbor.png)

### Forest Clearing
```
"a forest clearing with 20 trees surrounding a pond and scattered rocks"
```
![Forest](assets/forest.png)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/super-sniffle.git
cd super-sniffle
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Run
python main_v2.py
```

## Commands

| Command | Description |
|---------|-------------|
| `[prompt]` | Generate scene with auto-refinement |
| `/norefine` | Disable feedback loop for next prompt |
| `/refine` | Enable feedback loop (default) |
| `/iterations N` | Set max refinement iterations (default: 3) |
| `quit` | Exit |

## Feedback Loop

The Vision Critic (Stage 3) uses GPT-4o's vision capability to:

1. **Analyze** the rendered scene image
2. **Score** quality from 1-10
3. **Identify issues**: missing elements, positioning, scale, color, composition
4. **Generate fixes**: JSON patches for the scene
5. **Trigger refinement** if score < 8

The loop continues until the target score is reached or max iterations hit.

## Supported Entities

<details>
<summary><b>Buildings</b></summary>

- `house` - width, height, style (simple/modern), wall_color, roof_color
- `tower` - radius, height, battlements
- `castle` - width, depth, wall_height, tower_height
- `church` - width, depth, tower_height
- `lighthouse` - height, stripes, stripe_color

</details>

<details>
<summary><b>Nature</b></summary>

- `tree` - height, crown_style (natural/cone/sphere/layered)
- `bush` - radius, color
- `rock` - size, color (organic mesh with noise)
- `mountain` - base_radius, height, peaks (heightmap terrain)
- `pond` - radius (water surface with waves)
- `water` - width, depth

</details>

<details>
<summary><b>Objects</b></summary>

- `fountain` - radius, height, water_color
- `bench` - length, color
- `lamp_post` - height, light_color
- `bridge` - length, width, pillars
- `car` - length, body_color
- `boat` - length, hull_color

</details>

## How It Works

### Stage 1: LLM Scene Understanding
The LLM receives a prompt and outputs structured JSON with semantic entities:

```json
{
  "scene_description": "A small park with a pond",
  "ground": {"type": "plane", "width": 50, "color": "darkgreen"},
  "entities": [
    {"type": "pond", "radius": 8, "center": [0, 0, 0]},
    {"type": "tree", "height": 10, "center": [-15, 5, 0]},
    {"type": "bench", "length": 3, "center": [10, -5, 0]}
  ]
}
```

### Stage 2: Entity Decomposition
The `EntityAgent` converts each entity to primitives:

```python
# "house" entity becomes:
[
  {"type": "box", ...},      # walls
  {"type": "pyramid", ...},  # roof
  {"type": "box", ...},      # door
  {"type": "box", ...},      # windows
  {"type": "box", ...}       # chimney
]
```

### Hybrid Rendering
- **Terrain**: Heightmap with Perlin-like noise
- **Buildings**: Clean geometric primitives
- **Nature**: Organic procedural meshes (deformed icospheres, clustered spheres)

## Adding New Entities

```python
# In EntityAgent class:
@staticmethod
def _decompose_windmill(entity: dict) -> list:
    center = entity.get("center", [0, 0, 0])
    height = entity.get("height", 15)
    
    return [
        {"type": "cylinder", "radius": 2, "height": height, "center": center},
        {"type": "box", "width": 12, "height": 1, "depth": 0.5, 
         "center": [center[0], center[1], center[2] + height], "rotation": {"z": 45}}
    ]
```

## Tech Stack

- **OpenAI GPT-4o** - Scene understanding
- **PyVista** - 3D rendering
- **NumPy** - Procedural geometry

## License

MIT

---

*Built with curiosity and lots of `plotter.show()` calls* 🎨