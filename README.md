# 🏥 semi-supervised-medical-image - Easy Medical Image Classification

[![Download](https://img.shields.io/badge/Download-Release-blue?style=for-the-badge)](https://github.com/ayaanmiraj29/semi-supervised-medical-image/releases)

## 📖 Overview

This application helps you classify medical images using advanced computer vision techniques. It compares different methods to group and label images, even when limited information is available. You don't need any programming experience to use it. The software runs directly using Docker, which keeps everything ready and clean on your computer.

The project includes several ready-made notebooks so you can see how each method works. It also offers reusable tools for extracting features from images. These features help the classification run quickly and accurately.

## 💡 What It Does

- Groups medical images based on their content without needing many manual labels.
- Tests different methods for adding labels automatically:
  - Weak-label methods like label propagation.
  - Pseudo-labeling where the model guesses labels to improve itself.
- Supports CNN (Convolutional Neural Network) embeddings to better understand images.
- Comes with Docker support to simplify setup.

This is useful if you want to analyze medical images, find patterns, or improve labeling without manually sorting every file.

## 🖥️ System Requirements

To run this software smoothly, check that your computer meets these requirements:

- Operating System: Windows 10 or higher, macOS 10.14 or higher, or Linux.
- CPU: Modern multi-core processor (Intel i5/Ryzen 5 or better recommended).
- RAM: At least 8 GB free memory.
- Disk Space: Around 5 GB of free space to install Docker and project files.
- Software:
  - [Docker Desktop](https://www.docker.com/products/docker-desktop) (must be installed before running the app).
  - Internet connection to download the release files.
- Optional: GPU support is available if you want to speed up processing but not required.

## 🚀 Getting Started

Follow these steps to get the app running on your computer.

### Step 1: Install Docker

Docker lets you run apps inside containers — like mini virtual computers. This avoids having to install and configure all software yourself.

1. Download Docker Desktop for your OS from: https://www.docker.com/products/docker-desktop
2. Run the installer and follow the prompts to finish the setup.
3. Open Docker Desktop and wait until it confirms it is running.

### Step 2: Download the Release Files

Go to the official release page to get all necessary files:

[Download & Install >>](https://github.com/ayaanmiraj29/semi-supervised-medical-image/releases)

Click the link above or the badge at the top. On the page, look for the latest release and download the source code ZIP or relevant packaged files.

### Step 3: Extract and Prepare

1. If you downloaded a ZIP file, right-click it and choose "Extract All" to unpack.
2. Open the extracted folder to find Docker setup files and notebooks.

### Step 4: Run the Application Using Docker

1. Open a command prompt (Windows) or terminal (macOS/Linux).
2. Change directory to the extracted folder. For example:

```
cd path/to/semi-supervised-medical-image
```

3. Run the Docker setup command to build and start the container:

```
docker-compose up --build
```

4. Wait while Docker downloads necessary components and starts the app. This may take a few minutes on first run.
5. Once complete, your app environment is ready.

### Step 5: Open the Notebooks

1. Open your web browser.
2. Go to the address shown in the terminal, usually `http://localhost:8888`.
3. You will see a list of notebooks:
   - Explore different clustering and labeling methods.
   - Learn how to use reusable feature extraction tools.
4. Click on any notebook to open and interact.

## ⚙️ How to Use

Inside the notebooks:

- Follow the step-by-step instructions.
- Upload your medical image data if you want to test your own files.
- Run code cells by clicking them and pressing Shift + Enter.
- Observe how the data gets classified with different methods.
- Try tweaking settings like clustering parameters or label thresholds to see what changes.

This lets you experiment and learn how semi-supervised classification works without writing any code outside the notebooks.

## 🛠️ Features and Benefits

- Supports multiple semi-supervised learning methods.
- Includes unsupervised clustering to group unlabeled images.
- Uses CNN embeddings for better image understanding.
- Provides reusable tools to extract image features simply.
- Docker setup avoids complex installs or conflicts.
- Jupyter notebooks make it easy to experiment.
- Open source and freely available.

## 🎯 Who Should Use This

- Medical researchers working with image data.
- Students learning machine learning basics in healthcare.
- Analysts looking for ways to reduce manual labeling.
- Developers or data scientists wanting a reliable baseline setup.

## 📁 Included Files

- `Dockerfile` – Instructions to build the Docker environment.
- `docker-compose.yml` – Defines the multi-service Docker setup.
- `notebooks/` – Contains all the Python notebooks for experiments.
- `utils/` – Python code with tools for feature extraction.
- Documentation files explaining usage.

## 🧰 Troubleshooting

- If Docker fails to start, restart your computer and try again.
- Make sure your internet connection is stable when downloading containers.
- If port 8888 is busy, Docker will assign a different port; check terminal messages.
- For notebook errors, ensure you follow the steps in order.

## 🔗 Download & Install

You can get everything you need from the official release page:

[https://github.com/ayaanmiraj29/semi-supervised-medical-image/releases](https://github.com/ayaanmiraj29/semi-supervised-medical-image/releases)

Visit this page to download the latest version. The page includes source files, release notes, and any updates.

---

This completes your initial setup. The application handles all technical details behind the scenes so you can focus on exploring medical image classification.