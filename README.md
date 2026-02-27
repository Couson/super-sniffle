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

| # | Scene | Prompt | Initial Score | Best Score | Iterations | Entities | Key Issues Found |
|---|-------|--------|---------------|------------|------------|----------|------------------|
| 1 | Desert Oasis | "a desert oasis with palm trees, a small pond, and scattered rocks surrounded by sand dunes" | 4/10 | 5/10 | 3 | 9→12 | Missing pond, clustered trees, invisible rocks |
| 2 | Space Station | "a futuristic space station platform with cylindrical modules, a communication tower, and landing pads" | 7/10 | 7/10 | 3 | 6→15 | Floating tower, similar colors, sparse layout |
| 3 | Japanese Garden | "a serene Japanese garden with a small bridge over a pond, stone lanterns, and bonsai trees" | 5/10 | 5/10 | 3 | 9→18 | Missing lanterns & bonsai, floating bridge |
| 4 | Mountain Village | "a small mountain village with wooden cabins, a church, and pine trees on a hillside" | 4/10 | 5/10 | 3 | 7→10 | Missing cabins, missing pine trees, floating church |
| 5 | Coastal Harbor | "a coastal harbor with a lighthouse, wooden pier, boats, and a stone bridge" | 5/10 | 5/10 | 3 | 6→6 | Water not flush, boats floating, missing pier |

### Detailed Iteration Analysis

<details>
<summary><b>Desert Oasis</b></summary>

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 9 | 37 | 4/10 | Pond missing, trees clustered, rocks not visible |
| 2 | 10 | 38 | 5/10 | Ground color unrealistic, trees too small, rocks unnaturally placed |
| 3 | 12 | 49 | - | Scene refined with more elements |

**Critic Feedback Examples:**
- "The small pond mentioned in the request is not visible"
- "The trees seem clustered too closely together"
- "The rocks are missing or not visible"

</details>

<details>
<summary><b>Space Station</b></summary>

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 6 | 8 | 7/10 | Tower floating, colors too similar, scene sparse |
| 2 | 12 | 16 | 6/10 | Missing lamp posts, tower not centered, landing pads too small |
| 3 | 15 | 20 | - | Added more structural elements |

**Critic Feedback Examples:**
- "The communication tower seems to be floating above the platform"
- "Cylindrical modules and landing pads are too similar in color"
- "The scene appears sparse with too much empty space"

</details>

<details>
<summary><b>Japanese Garden</b></summary>

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 9 | 30 | 5/10 | Missing stone lanterns, missing bonsai, bridge floating |
| 2 | 15 | 60 | 5/10 | Bonsai still missing, bushes clustered, ground too large |
| 3 | 18 | 95 | - | Significantly more detail added |

**Critic Feedback Examples:**
- "The scene is missing stone lanterns"
- "The scene is missing bonsai trees"
- "The bridge appears to be floating above the pond"

</details>

<details>
<summary><b>Mountain Village</b></summary>

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 7 | 49 | 4/10 | Missing cabins, missing pine trees, church floating |
| 2 | 9 | 69 | 5/10 | Not enough trees, church floating, mountain too small |
| 3 | 10 | 75 | - | Added more buildings and trees |

**Critic Feedback Examples:**
- "Missing wooden cabins that were specified in the scene"
- "Missing pine trees that were specified in the scene"
- "Church placement suggests it is floating on the mountain"

</details>

<details>
<summary><b>Coastal Harbor</b></summary>

| Iteration | Entities | Meshes | Score | Issues Identified |
|-----------|----------|--------|-------|-------------------|
| 1 | 6 | 20 | 5/10 | Water not flush with ground, bridge elevated, boats floating |
| 2 | 6 | 22 | 4/10 | Water missing, pier not visible, lighthouse misplaced |
| 3 | 6 | 22 | - | Minor adjustments made |

**Critic Feedback Examples:**
- "The water is not flush with the ground plane"
- "Boats are floating above the water rather than on it"
- "The wooden pier is not visible despite being requested"

</details>

### Key Observations

1. **Feedback Loop Effectiveness**: The vision critic successfully identifies missing elements, positioning issues, and scale problems across all test scenes.

2. **Issue Categories Detected**:
   | Issue Type | Frequency | Description |
   |------------|-----------|-------------|
   | Missing | High | Elements from prompt not rendered |
   | Position | High | Objects floating or incorrectly placed |
   | Scale | Medium | Size mismatches between objects |
   | Color | Medium | Colors not matching expectations |
   | Composition | Low | Overall layout problems |

3. **Entity Growth**: The feedback loop consistently adds entities (9→12, 6→15, 9→18) as the critic identifies missing elements.

4. **Mesh Complexity**: More iterations = more detailed scenes (30→95 meshes for Japanese Garden).

5. **Limitations Observed**:
   - Some specialized entities (bonsai, stone lanterns) lack decomposition rules
   - Z-positioning (floating objects) is a recurring challenge
   - Color naming inconsistencies (e.g., "sandy_brown" vs "sand")

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