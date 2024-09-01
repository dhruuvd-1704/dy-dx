# **SIH 2024**

**Team Name: dy/dx**

## Overview

Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. This project leverages the XceptionNet architecture to detect such deepfakes with high accuracy.

## Problem Statement

- **Problem Statement ID:** 1683
- **Problem Statement Title:** Development of AI/ML based solution for detection of face-swap based deep fake videos

## Demo Video
[Watch the video](https://drive.google.com/file/d/VIDEO_ID/view)


https://github.com/user-attachments/assets/d2787c38-ca6b-4b32-a4f4-93391cd2936d



# Deepfake Detection Using XceptionNet CNN MODEL

[![DOI](https://img.shields.io/badge/DOI-10.1109/RASSE60029.2023.10363477-blue)](https://doi.org/10.1109/RASSE60029.2023.10363477)

This repository contains the implementation of the deepfake detection model using XceptionNet as presented in the paper: **A. V and P. T. Joy, "Deepfake Detection Using XceptionNet," 2023 IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE), Kerala, India, 2023, pp. 1-5.** [Read the paper](https://doi.org/10.1109/RASSE60029.2023.10363477)

## Future Aspects

1. **Browser Extension**: Develop a browser extension that can detect deepfake content directly within the browser, providing feedback to users as they browse the web.

2. **Social Media Links Support**: Integrate support for analyzing and detecting deepfakes in videos shared on social media platforms with direct links.

## Model Files

You can download the trained models from the following link:

[Download Model](https://drive.google.com/file/d/1jy6vLdn9RCrImWIL_3y4UGdcmuF85Pg4/view?usp=sharing)

## Installation

Clone this repository:

```bash
git clone https://github.com/dhruuvd-1704/dy-dx.git
cd dy-dx
```

Create a Terminal for Backend and run:
```bash
uvicorn backend:app --reload
```

Create a new Terminal for Frontend and run:
```bash
streamlit run app.py
```

**Team:<br/>**
Dhruv Desai<br/>
Atharva Humane<br/>
Niranjan More<br/>
Mithilesh Singh<br/>
Vaishnavi Hud<br/>
Kasturi Pawar<br/>
