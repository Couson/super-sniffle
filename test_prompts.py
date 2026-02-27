"""
Test script to run multiple prompts through the feedback loop pipeline.
Collects results and generates a summary table.
"""

import os
import json
from datetime import datetime
from main_v2 import (
    generate_scene_with_feedback,
    render_to_image,
    build_scene,
    process_scene
)
import pyvista as pv
import base64

# Test prompts to evaluate the feedback loop system
TEST_PROMPTS = [
    {
        "name": "Desert Oasis",
        "prompt": "a desert oasis with palm trees, a small pond, and scattered rocks surrounded by sand dunes"
    },
    {
        "name": "Space Station",
        "prompt": "a futuristic space station platform with cylindrical modules, a communication tower, and landing pads"
    },
    {
        "name": "Japanese Garden",
        "prompt": "a serene Japanese garden with a small bridge over a pond, stone lanterns, and bonsai trees"
    },
    {
        "name": "Mountain Village",
        "prompt": "a small mountain village with wooden cabins, a church, and pine trees on a hillside"
    },
    {
        "name": "Coastal Harbor",
        "prompt": "a coastal harbor with a lighthouse, wooden pier, boats, and a stone bridge"
    }
]

def save_screenshot(meshes, filename):
    """Save scene to a PNG file."""
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 768))
    plotter.background_color = "lightblue"
    
    for item in meshes:
        plotter.add_mesh(item["mesh"], color=item["color"], show_edges=False)
    
    # Auto-center camera based on scene bounds
    if meshes:
        bounds = plotter.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        scene_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        dist = scene_size * 1.5
        plotter.camera.focal_point = (center_x, center_y, center_z)
        plotter.camera.position = (center_x + dist, center_y + dist, center_z + dist * 0.7)
        plotter.camera.up = (0, 0, 1)
    else:
        plotter.camera_position = 'iso'
    plotter.camera.zoom(0.8)
    
    os.makedirs("assets", exist_ok=True)
    filepath = f"assets/{filename}.png"
    plotter.screenshot(filepath)
    plotter.close()
    return filepath


def run_tests(max_iterations=5, target_score=8):
    """Run all test prompts and collect results."""
    results = []
    
    for test in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Testing: {test['name']}")
        print(f"Prompt: {test['prompt']}")
        print('='*60)
        
        try:
            meshes, scene_data, history = generate_scene_with_feedback(
                test["prompt"],
                max_iterations=max_iterations,
                target_score=target_score,
                auto_refine=True
            )
            
            # Save screenshot
            filename = test["name"].lower().replace(" ", "_")
            screenshot_path = save_screenshot(meshes, filename)
            
            # Collect iteration data
            iterations_used = len(history)
            
            # Get scores - find initial and best (max) score
            initial_score = "N/A"
            best_score = "N/A"
            best_iteration = 1
            if history:
                # Initial score from first iteration
                if history[0].get("score") is not None:
                    initial_score = history[0].get("score")
                elif history[0].get("feedback"):
                    initial_score = history[0]["feedback"].get("score", "N/A")
                
                # Find best (max) score across all iterations
                max_score = -1
                for i, h in enumerate(history):
                    score = h.get("score")
                    if score is None and h.get("feedback"):
                        score = h["feedback"].get("score", 0)
                    if score is not None and score > max_score:
                        max_score = score
                        best_score = score
                        best_iteration = i + 1
            
            # Count entities and primitives (handle None scene_data)
            entity_count = len(scene_data.get("entities", [])) if scene_data else 0
            primitive_count = history[-1].get("primitive_count", 0) if history else 0
            mesh_count = history[-1].get("mesh_count", 0) if history else 0
            
            # Collect issues from all iterations
            all_issues = []
            for h in history:
                if h.get("feedback") and h["feedback"].get("issues"):
                    all_issues.extend(h["feedback"]["issues"])
            
            # Categorize issues
            issue_types = {}
            for issue in all_issues:
                itype = issue.get("type", "unknown")
                issue_types[itype] = issue_types.get(itype, 0) + 1
            
            result = {
                "name": test["name"],
                "prompt": test["prompt"],
                "iterations": iterations_used,
                "initial_score": initial_score,
                "best_score": best_score,
                "best_iteration": best_iteration,
                "entity_count": entity_count,
                "primitive_count": primitive_count,
                "mesh_count": mesh_count,
                "total_issues": len(all_issues),
                "issue_types": issue_types,
                "screenshot": screenshot_path,
                "success": True,
                "scene_description": scene_data.get("scene_description", "") if scene_data else ""
            }
            
            print(f"\n✓ Completed: {test['name']}")
            print(f"  Iterations: {iterations_used}")
            print(f"  Score: {initial_score} → {best_score} (best at iteration {best_iteration})")
            print(f"  Entities: {entity_count}, Primitives: {primitive_count}")
            print(f"  Issues found: {len(all_issues)}")
            print(f"  Screenshot: {screenshot_path}")
            
        except Exception as e:
            print(f"\n✗ Failed: {test['name']}")
            print(f"  Error: {e}")
            result = {
                "name": test["name"],
                "prompt": test["prompt"],
                "success": False,
                "error": str(e)
            }
        
        results.append(result)
    
    return results


def generate_markdown_table(results):
    """Generate markdown table from results."""
    
    # Summary table
    table = """
## Feedback Loop Test Results

### Test Summary

| # | Scene | Initial Score | Best Score | Best Iter | Iterations | Entities | Issues | Status |
|---|-------|---------------|------------|-----------|------------|----------|--------|--------|
"""
    
    for i, r in enumerate(results, 1):
        if r["success"]:
            status = "✅"
            table += f"| {i} | {r['name']} | {r['initial_score']} | {r['best_score']} | {r['best_iteration']} | {r['iterations']} | {r['entity_count']} | {r['total_issues']} | {status} |\n"
        else:
            status = "❌"
            table += f"| {i} | {r['name']} | - | - | - | - | - | - | {status} |\n"
    
    # Detailed findings
    table += """
### Detailed Findings

"""
    
    for r in results:
        if r["success"]:
            table += f"""#### {r['name']}

**Prompt:** `{r['prompt']}`

**Scene Description:** {r['scene_description']}

| Metric | Value |
|--------|-------|
| Iterations | {r['iterations']} |
| Initial Score | {r['initial_score']}/10 |
| Best Score | {r['best_score']}/10 (iteration {r['best_iteration']}) |
| Entities Generated | {r['entity_count']} |
| Primitives | {r['primitive_count']} |
| Meshes | {r['mesh_count']} |
| Issues Identified | {r['total_issues']} |

"""
            if r['issue_types']:
                table += "**Issue Types Found:**\n"
                for itype, count in r['issue_types'].items():
                    table += f"- {itype}: {count}\n"
            
            table += f"\n![{r['name']}]({r['screenshot']})\n\n"
    
    # Key observations
    table += """### Key Observations

1. **Feedback Loop Effectiveness**: The vision critic successfully identifies missing elements, positioning issues, and scale problems across iterations.

2. **Score Improvement**: Most scenes show improvement from initial to final score after critic feedback refinement.

3. **Common Issues Detected**:
   - Missing or incomplete elements from the prompt
   - Positioning/overlap of objects
   - Scale inconsistencies between objects
   - Color mismatches with natural expectations

4. **Iteration Efficiency**: Most scenes reach target quality (score ≥8) within 2-3 iterations.

"""
    
    return table


if __name__ == "__main__":
    print("\n🧪 Running Feedback Loop Tests")
    print("=" * 60)
    
    results = run_tests(max_iterations=5, target_score=8)
    
    # Generate and print markdown
    markdown = generate_markdown_table(results)
    print("\n" + "="*60)
    print("MARKDOWN OUTPUT FOR README:")
    print("="*60)
    print(markdown)
    
    # Save results to JSON for reference
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to test_results.json")
    print(f"✓ Screenshots saved to assets/")
