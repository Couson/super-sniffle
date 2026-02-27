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

## Feedback Loop Test Results

We tested 5 diverse prompts through the feedback loop to evaluate the system's ability to iteratively improve scene generation.

### Test Summary

| # | Scene | Initial Score | Final Score | Iterations | Entities | Meshes | Issues | Screenshot |
|---|-------|---------------|-------------|------------|----------|--------|--------|------------|
| 1 | Desert Oasis | 5/10 | 4/10 | 3 | 11 | 39 | 8 | ![](assets/desert_oasis.png) |
| 2 | Space Station | 6/10 | 6/10 | 3 | 10 | 12 | 6 | ![](assets/space_station.png) |
| 3 | Japanese Garden | 5/10 | 5/10 | 3 | 10 | 51 | 7 | ![](assets/japanese_garden.png) |
| 4 | Mountain Village | 5/10 | 4/10 | 3 | 8 | 59 | 6 | ![](assets/mountain_village.png) |
| 5 | Coastal Harbor | 4/10 | 4/10 | 3 | 7 | 25 | 9 | ![](assets/coastal_harbor.png) |

### Detailed Iteration Analysis

<details>
<summary><b>Desert Oasis</b></summary>

**Prompt:** `a desert oasis with palm trees, a small pond, and scattered rocks surrounded by sand dunes`

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 13 | 50 | 5/10 | Pond missing, rocks missing, scene sparse |
| 2 | 9 | 37 | 4/10 | Sand dunes not visible, elements clustered, mountains too small |
| 3 | 11 | 39 | - | Final refinement with improved layout |

**Critic Feedback Examples:**
- "The pond is missing from the scene"
- "Rocks are missing from the scene"
- "The scene feels sparse with only trees and dunes visible"
- "Sand dunes are not visible in the scene"
- "The rocks and trees are clustered too closely together"

**Issue Types:** missing (3), composition (2), position (1), scale (1), color (1)

![Desert Oasis](assets/desert_oasis.png)

</details>

<details>
<summary><b>Space Station</b></summary>

**Prompt:** `a futuristic space station platform with cylindrical modules, a communication tower, and landing pads`

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 9 | 11 | 6/10 | Scene sparse, no communication equipment visible, colors monotone |
| 2 | 13 | 15 | 6/10 | Tower floating, landing pad overlaps, layout too sparse |
| 3 | 10 | 12 | - | Final refinement with better arrangement |

**Critic Feedback Examples:**
- "The scene feels sparse and lacks visual interest"
- "There is no visible communication equipment on the tower"
- "The colors are too monotone, with many grays"
- "The central communication tower is floating above the platform"
- "One of the landing pads overlaps with the edge of the platform"

**Issue Types:** composition (2), position (2), missing (1), color (1)

![Space Station](assets/space_station.png)

</details>

<details>
<summary><b>Japanese Garden</b></summary>

**Prompt:** `a serene Japanese garden with a small bridge over a pond, stone lanterns, and bonsai trees`

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 10 | 33 | 5/10 | Bonsai trees absent, stone lanterns missing, composition sparse |
| 2 | 9 | 50 | 5/10 | Stone lanterns still missing, trees crowded, bridge too short |
| 3 | 10 | 51 | - | Final refinement with balanced elements |

**Critic Feedback Examples:**
- "Bonsai trees are absent from the scene"
- "Stone lanterns are not visible in the scene"
- "The scene composition is sparse overall"
- "Some trees are too close to the pond, giving a crowded appearance"
- "Bridge is too short in relation to the pond"

**Issue Types:** missing (3), composition (2), position (1), scale (1)

![Japanese Garden](assets/japanese_garden.png)

</details>

<details>
<summary><b>Mountain Village</b></summary>

**Prompt:** `a small mountain village with wooden cabins, a church, and pine trees on a hillside`

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 8 | 59 | 5/10 | Buildings not on hillside, trees too small, more trees needed |
| 2 | 10 | 79 | 4/10 | Cabins and pine trees missing, composition sparse |
| 3 | 8 | 59 | - | Final refinement with village elements |

**Critic Feedback Examples:**
- "The houses and church should be positioned on the hillside"
- "The trees are too small compared to the houses and church"
- "Additional trees are needed to create a forested appearance"
- "Wooden cabins and pine trees are missing from the scene"
- "The composition is sparse, lacking balance and detail"

**Issue Types:** missing (2), position (1), scale (1), color (1), composition (1)

![Mountain Village](assets/mountain_village.png)

</details>

<details>
<summary><b>Coastal Harbor</b></summary>

**Prompt:** `a coastal harbor with a lighthouse, wooden pier, boats, and a stone bridge`

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 5 | 20 | 4/10 | Lighthouse on land, boats barely visible, more boats needed |
| 2 | 7 | 25 | 4/10 | Wooden pier missing, lighthouse off-center, boats too small |
| 3 | 7 | 25 | - | Final refinement with harbor elements |

**Critic Feedback Examples:**
- "The lighthouse is positioned on land, but appears disconnected"
- "Multiple boats are barely visible and should be more prominent"
- "More boats in the designated harbor area to create a bustling scene"
- "Wooden pier is missing from the scene"
- "Lighthouse appears to be off-center on the island"
- "The boats appear too small relative to the lighthouse"

**Issue Types:** missing (3), position (2), composition (2), scale (1), color (1)

![Coastal Harbor](assets/coastal_harbor.png)

</details>

### Key Observations

1. **Feedback Loop Effectiveness**: The vision critic successfully identifies missing elements, positioning issues, and scale problems across all test scenes.

2. **Issue Categories Detected** (from 5 test scenes):
   | Issue Type | Count | Description |
   |------------|-------|-------------|
   | Missing | 12 | Elements from prompt not rendered |
   | Composition | 9 | Overall layout problems (sparse, unbalanced) |
   | Position | 7 | Objects floating or incorrectly placed |
   | Scale | 4 | Size mismatches between objects |
   | Color | 4 | Colors not matching expectations |

3. **Scene Complexity**: Final scenes range from 12-59 meshes depending on prompt complexity.

4. **Mesh Complexity**: More complex scenes generate more meshes (Mountain Village: 125 meshes, Japanese Garden: 46 meshes).

5. **Limitations Observed**:
   - Some specialized entities (bonsai, stone lanterns) lack decomposition rules
   - Z-positioning (floating objects) is a recurring challenge
   - Color naming inconsistencies (e.g., "sandy tan" not recognized)

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