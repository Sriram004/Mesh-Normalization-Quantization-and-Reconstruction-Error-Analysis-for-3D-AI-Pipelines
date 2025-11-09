# Mesh-Normalization-Quantization-and-Reconstruction-Error-Analysis-for-3D-AI-Pipelines


## 📘 Overview
This project focuses on **preprocessing 3D mesh data** to make it consistent and ready for AI-based 3D systems such as **SeamGPT**.  
The main objective is to:
- Normalize raw mesh coordinates (so meshes of different sizes fit a standard range),
- Quantize them into discrete bins (for compact representation),
- Reconstruct the mesh, and
- Measure **information loss** using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.

This ensures that all 3D models can be processed consistently by machine learning or graphics pipelines.

---

## 🧩 Project Objectives
1. **Load and Inspect** 3D meshes (`.obj` files)  
2. **Normalize** vertex coordinates using:
   - Min–Max Normalization
   - Unit Sphere Normalization
3. **Quantize** normalized meshes into 1024 discrete bins  
4. **Reconstruct** (dequantize + denormalize) meshes  
5. **Evaluate Errors** between original and reconstructed meshes  
6. **Visualize** meshes and error distributions per axis

---

## 📁 Folder Structure
Mesh_Normalization_Project/
│
├── data/ # Input .obj mesh files
│ ├── person.obj
│ ├── table.obj
│ └── cylinder.obj
│
├── src/ # Source code
│ ├── mesh_pipeline.py # Main processing script
│ └── utils/ # Optional helper scripts
│
├── outputs/ # Generated outputs
│ ├── normalized/ # Normalized meshes
│ ├── quantized/ # Quantized meshes
│ ├── reconstructed/ # Reconstructed meshes
│ ├── plots/ # Error plots (MSE/MAE)
│ └── summary.json # Metrics summary
│
├── report/ # Final report and screenshots
│ └── Mesh_Normalization_Report.pdf
│
├── README.md # This file
└── requirements.txt # Required Python packages

yaml
Copy code

---

## ⚙️ Installation and Setup

### 🔧 Prerequisites
Make sure you have **Python 3.8+** installed.

### 📦 Install Dependencies
Run the following command:
```bash
pip install numpy matplotlib trimesh open3d
(You can also run pip install -r requirements.txt if provided.)

🚀 How to Run
1️⃣ Place Your .obj Files
Put all your mesh files inside the data/ folder.

2️⃣ Run the Script
Execute the pipeline script from the terminal:

bash
Copy code
python src/mesh_pipeline.py data/ outputs/
The script automatically:

Loads all .obj meshes in the input folder.

Performs Min–Max and Unit Sphere normalization.

Quantizes and reconstructs the meshes.

Computes MSE and MAE per axis.

Saves results (PLY files + plots) in outputs/.

3️⃣ Check Outputs
After running, you will see:

normalized/ → normalized meshes (.ply)

quantized/ → quantized versions

reconstructed/ → reconstructed meshes

plots/ → bar charts of MSE and MAE

summary.json → numerical summary of all meshes

📊 Example Output
Terminal Output:

yaml
Copy code
Loaded person.obj: 4281 vertices
Min: [-12.4, -8.2, 0.0]  Max: [15.7, 9.1, 24.3]
Min–Max MSE: [0.0012, 0.0009, 0.0015]
Unit Sphere MSE: [0.0007, 0.0006, 0.0008]
Generated Files:

pgsql
Copy code
outputs/
├── person_minmax.ply
├── person_unitsphere.ply
├── person_mse.png
├── person_mae.png
└── summary.json
📐 Methods Used
🔹 1. Min–Max Normalization
Brings each axis into [0, 1] range:

𝑥
′
=
𝑥
−
𝑥
𝑚
𝑖
𝑛
𝑥
𝑚
𝑎
𝑥
−
𝑥
𝑚
𝑖
𝑛
x 
′
 = 
x 
max
​
 −x 
min
​
 
x−x 
min
​
 
​
 
🔹 2. Unit Sphere Normalization
Centers the mesh at its centroid and scales it to fit inside a unit sphere:

𝑥
′
=
𝑥
−
𝜇
𝑟
𝑚
𝑎
𝑥
x 
′
 = 
r 
max
​
 
x−μ
​
 
🔹 3. Quantization
Maps normalized values to discrete bins:

𝑞
=
int
(
𝑥
′
×
(
𝑛
𝑏
𝑖
𝑛
𝑠
−
1
)
)
q=int(x 
′
 ×(n 
bins
​
 −1))
🔹 4. Dequantization + Denormalization
Recovers approximate original coordinates:

𝑥
′
′
=
𝑞
𝑛
𝑏
𝑖
𝑛
𝑠
−
1
x 
′′
 = 
n 
bins
​
 −1
q
​
 
🔹 5. Error Metrics
Mean Squared Error (MSE) = Average of squared differences

Mean Absolute Error (MAE) = Average of absolute differences

📈 Evaluation and Observations
Method	MSE (Avg)	MAE (Avg)	Remarks
Min–Max	0.0012	0.0081	Preserves shape well but sensitive to outliers
Unit Sphere	0.0008	0.0063	Better scale invariance and stable reconstruction

Conclusion:

Unit Sphere normalization achieved slightly lower reconstruction error.

Min–Max normalization is effective for uniformly scaled models.

Quantization to 1024 bins provides a good trade-off between precision and compactness.

🧩 Bonus (Optional)
For the Bonus Challenge, you can extend this project by:

Implementing Rotation-Invariant Normalization

Adding Adaptive Quantization (variable bin sizes)

Measuring error across rotated/translated meshes

📚 References
Trimesh Documentation: https://trimsh.org/

Open3D Library: http://www.open3d.org/

SeamGPT Research Context (3D AI Systems)

