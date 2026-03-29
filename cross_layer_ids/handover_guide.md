# 🛡️ Hackathon Handover Guide: Cross-Layer Vehicle IDS

This guide provides the final checklist and strategic assets to ensure a successful demonstration and defense of your project.

## 1. Hardware Checkout (MANDATORY)
Before entering the venue, verify your embedded pipeline:
- [ ] **Flash**: Open `esp32/cross_layer_ids/cross_layer_ids.ino` and upload to your **ESP32 Dev Module**.
- [ ] **Serial**: Open Serial Monitor (115200) and verify "CrossLayer-IDS v2.0 Ready" appears.
- [ ] **Wiring**: 
    - GPIO 2 → Green (SAFE)
    - GPIO 4 → Yellow (SINGLE)
    - GPIO 5 → Red (COORD)
- [ ] **Cable**: Use a **USB Data Cable** (not charge-only).

## 2. Live Demo Script (The "3-Minute Flow")
| Time | Action | What to Say |
| :--- | :--- | :--- |
| **0:00** | Start `dashboard.py` | "We are monitoring a software-defined vehicle's sensor, CAN, and V2X layers in real-time." |
| **0:45** | Click **GPS Spoof** | "A single-layer attack is detected immediately by raw features. Note the Yellow LED." |
| **1:15** | Click **Coord (All 3)** | "Now we inject a synchronized attack across all layers. Each layer looks normal, but our **Physics-Consistency Engine** detects the violation." |
| **2:00** | Scroll to **Baselines** | "Watch the 'Misclassified' status on raw-only baselines. They fail because they lack cross-layer awareness." |
| **2:45** | Show **Docs/Plots** | "These diagnostic plots show our model maintains detection even when the attacker stays within plausible noise ranges." |

## 3. Defending the Numbers (Judge Q&A)
> [!IMPORTANT]
> Be honest and technically grounded. Judges value rigor over inflated metrics.

- **Q: Why 77% F1?**
    - **A:** "Coordinated attacks are designed to be stealthy. 77% detection is a massive leap over the 0-10% achieved by single-layer systems on these specific patterns."
- **Q: Why only 4.5% improvement?**
    - **A:** "While Random Forest can implicitly learn some relationships, our explicit physics features provide **interpretability**. In automotive safety, we can explain *why* an attack was flagged (e.g., 'Bicycle model yaw vs IMU discrepancy')."
- **Q: Is the data realistic?**
    - **A:** "We use a standard bicycle model with noise parameters modeled from real vehicle sensors. It's the most rigorous way to test cross-layer logic without a $100k test rig."

## 4. Winning Assets in `docs/`
- **`plot4_comparison.png`**: Your "Winning Slide." Shows the failure of single-layer models vs. your success.
- **`plot1_features.png`**: The "Science Slide." Demonstrates the physical separation of attack classes.

## 5. Nuclear Fallback
If the hardware fails at the venue:
1.  Run the streamer in **Simulation Mode** (it will auto-detect no ESP32).
2.  Open the pre-recorded `demo_backup.mp4` on your desktop.
3.  Display the 4 static PNGs from the `docs/` folder.

**Good luck at the Sensathon! You have a technically solid, defensible project.**
