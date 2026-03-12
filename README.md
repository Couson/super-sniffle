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
| 1 | Gemini (LLM) | Scene understanding → high-level entities with parameters |
| 2 | EntityAgent (Rules) | **Deterministic** decomposition → geometric primitives (no LLM) |
| 3 | Vision Critic (LLM) | Analyze rendered image → score & feedback → refinement loop |

**Why three stages?**
- **Stage 1**: LLM focuses on *what* to create, not *how* to build it
- **Stage 2**: Entity decomposition is **deterministic** — no LLM calls, just rules
- **Stage 3**: Vision model validates output and triggers automatic refinement
- Self-improving pipeline that iterates until quality target is met

## Examples

### Desert Oasis
```
"a desert oasis with palm trees, a small pond, and scattered rocks surrounded by sand dunes"
```
![Desert Oasis](assets/desert_oasis.png)

### Space Station
```
"a futuristic space station platform with cylindrical modules, a communication tower, and landing pads"
```
![Space Station](assets/space_station.png)

### Japanese Garden
```
"a serene Japanese garden with a small bridge over a pond, stone lanterns, and bonsai trees"
```
![Japanese Garden](assets/japanese_garden.png)

### Mountain Village
```
"a small mountain village with wooden cabins, a church, and pine trees on a hillside"
```
![Mountain Village](assets/mountain_village.png)

### Coastal Harbor
```
"a coastal harbor with a lighthouse, wooden pier, boats, and a stone bridge"
```
![Coastal Harbor](assets/coastal_harbor.png)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/super-sniffle.git
cd super-sniffle

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Set your API keys
echo "GOOGLE_API_KEY=your-google-key" >> .env

# Run
uv run python main_v2.py
```

## Commands

| Command | Description |
|---------|-------------|
| `[prompt]` | Generate scene with auto-refinement (keeps best score) |
| `/norefine` | Disable feedback loop for next prompt |
| `/refine` | Enable feedback loop (default) |
| `/iterations N` | Set max refinement iterations (default: 5) |
| `quit` | Exit |

## Feedback Loop

The Vision Critic (Stage 3) uses Gemini's vision capability to:

1. **Analyze** the rendered scene image
2. **Score** quality from 1-10
3. **Identify issues**: missing elements, positioning, scale, color, composition
4. **Generate fixes**: JSON patches for the scene
5. **Trigger refinement** if score < 8

The loop continues until the target score is reached or max iterations hit.

## Feedback Loop Test Results

We tested 5 diverse prompts through the feedback loop to evaluate the system's ability to iteratively improve scene generation. **The system keeps the best-scoring scene across all iterations.**

### Test Summary

| # | Scene | Initial | Best | Best Iter | Iterations | Entities | Screenshot |
|---|-------|---------|------|-----------|------------|----------|------------|
| 1 | Desert Oasis | 6/10 | 6/10 | 1 | 5 | 13 | ![](assets/desert_oasis.png) |
| 2 | Space Station | 6/10 | 7/10 | 5 | 5 | 11 | ![](assets/space_station.png) |
| 3 | Japanese Garden | 6/10 | 6/10 | 1 | 5 | 10 | ![](assets/japanese_garden.png) |
| 4 | Mountain Village | 5/10 | 5/10 | 1 | 5 | 7 | ![](assets/mountain_village.png) |
| 5 | Coastal Harbor | 5/10 | 5/10 | 1 | 5 | 6 | ![](assets/coastal_harbor.png) |

**Key: Best Iter** = Iteration with highest score (system returns this scene)

### Detailed Iteration Analysis

<details>
<summary><b>Desert Oasis</b> - Best at iteration 1 (score 6)</summary>

**Prompt:** `a desert oasis with palm trees, a small pond, and scattered rocks surrounded by sand dunes`

| Iteration | Score | Result |
|-----------|-------|--------|
| 1 | **6/10** | ★ Best - Oasis with trees, pond, and dunes |
| 2 | 3/10 | Regression - Mountains too dominant |
| 3 | 4/10 | Gray ground instead of sandy |
| 4 | 5/10 | Sparse, dunes not surrounding |
| 5 | 3/10 | Color issues persist |

**Critic Feedback:**
- "The mountains surrounding the oasis are disproportionately small"
- "The pond is positioned off-center"
- "Ground and mountains are gray instead of sandy brown"

![Desert Oasis](assets/desert_oasis.png)

</details>

<details>
<summary><b>Space Station</b> - Score improved: 6 → 7 (best at iteration 5)</summary>

**Prompt:** `a futuristic space station platform with cylindrical modules, a communication tower, and landing pads`

| Iteration | Score | Result |
|-----------|-------|--------|
| 1 | 6/10 | Basic layout, communication tower missing |
| 2 | 6/10 | Modules floating above platform |
| 3 | 6/10 | Scale issues persist |
| 4 | 6/10 | Similar issues |
| 5 | **7/10** | ★ Best - Improved positioning and density |

**Critic Feedback:**
- "Cylindrical modules are floating above the platform"
- "Ground color is too stark"
- "Scene feels sparse and lacks visual coherence"

![Space Station](assets/space_station.png)

</details>

<details>
<summary><b>Japanese Garden</b> - Best at iteration 1 (score 6)</summary>

**Prompt:** `a serene Japanese garden with a small bridge over a pond, stone lanterns, and bonsai trees`

| Iteration | Score | Result |
|-----------|-------|--------|
| 1 | **6/10** | ★ Best - Bridge, pond, and basic layout |
| 2 | 4/10 | Bridge not over pond properly |
| 3 | 5/10 | Minor improvements |
| 4 | 4/10 | Stone lanterns still not visible |
| 5 | 4/10 | Persistent issues |

**Critic Feedback:**
- "Bridge does not appear to connect to the pond area"
- "Stone lanterns are missing"
- "The scene needs more elements to feel like a garden"

![Japanese Garden](assets/japanese_garden.png)

</details>

<details>
<summary><b>Mountain Village</b> - Consistent at iteration 1 (score 5)</summary>

**Prompt:** `a small mountain village with wooden cabins, a church, and pine trees on a hillside`

| Iteration | Score | Result |
|-----------|-------|--------|
| 1 | **5/10** | ★ Best - Village with mountain backdrop |
| 2 | 5/10 | Mountain too dominant |
| 3 | 4/10 | Cabins missing, sparse layout |
| 4 | 4/10 | Mountain overwhelms village |
| 5 | 5/10 | Similar to iteration 1 |

**Critic Feedback:**
- "The mountain is positioned too close to the edge"
- "Trees are sparsely positioned and isolated"
- "The mountain's scale is too large, overwhelming village features"

![Mountain Village](assets/mountain_village.png)

</details>

<details>
<summary><b>Coastal Harbor</b> - Best at iteration 1 (score 5)</summary>

**Prompt:** `a coastal harbor with a lighthouse, wooden pier, boats, and a stone bridge`

| Iteration | Score | Result |
|-----------|-------|--------|
| 1 | **5/10** | ★ Best - Lighthouse, boats, bridge present |
| 2 | 5/10 | Water still not visible |
| 3 | 4/10 | Pier missing, boats too small |
| 4 | 4/10 | Boats on green plane not water |
| 5 | 4/10 | Persistent water issues |

**Critic Feedback:**
- "The wooden pier is missing from the scene"
- "The water is not visible"
- "Boats are floating on a green plane instead of water"

![Coastal Harbor](assets/coastal_harbor.png)

</details>

### Key Observations

1. **Best Score Preservation**: The system keeps the highest-scoring scene across iterations. Most test cases (4/5) had their best score at iteration 1, with later iterations often regressing.

2. **Space Station Improvement**: The only scene to improve was Space Station (6→7), showing the feedback loop can help when the critic provides actionable fixes.

3. **Issue Categories Detected** (from 5 test scenes):
   | Issue Type | Count | Description |
   |------------|-------|-------------|
   | Position | 26 | Objects floating, overlapping, or misplaced |
   | Missing | 15 | Elements from prompt not rendered |
   | Composition | 11 | Layout too sparse or unbalanced |
   | Scale | 7 | Size mismatches between objects |
   | Color | 8 | Colors not matching expectations (e.g., "sandy brown") |

4. **Limitations Observed**:
   - Color naming: "sandy brown", "sandybrown", "metallic gray" not in color map
   - Water rendering: Coastal/harbor scenes struggle with water planes
   - Scale balance: Mountains often dominate village elements
   - Z-positioning: Objects floating above ground is recurring

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

- **Google Gemini** - Scene understanding & vision
- **PyVista** - 3D rendering
- **NumPy** - Procedural geometry

## License

MIT

---

*Built with curiosity and lots of `plotter.show()` calls* 🎨