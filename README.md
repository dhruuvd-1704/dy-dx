# **SIH 2024**

**Team Name: dy/dx**

## Overview

Deepfakes are synthetic media in which a person in an existing image or video is replaced with someone else's likeness. This project leverages the XceptionNet architecture to detect such deepfakes with high accuracy.

## Problem Statement

- **Problem Statement ID:** 1683
- **Problem Statement Title:** Development of AI/ML based solution for detection of face-swap based deep fake videos

## Demo Video
[Watch the video](https://drive.google.com/file/d/VIDEO_ID/view)


https://github.com/user-attachments/assets/9845d7f2-f4dc-4547-9186-aa3bcc5676ad





# Deepfake Detection Using XceptionNet CNN MODEL

[![DOI](https://img.shields.io/badge/DOI-10.1109/RASSE60029.2023.10363477-blue)](https://doi.org/10.1109/RASSE60029.2023.10363477)

This repository contains the implementation of the deepfake detection model using XceptionNet as presented in the paper: **A. V and P. T. Joy, "Deepfake Detection Using XceptionNet," 2023 IEEE International Conference on Recent Advances in Systems Science and Engineering (RASSE), Kerala, India, 2023, pp. 1-5.** [Read the paper](https://doi.org/10.1109/RASSE60029.2023.10363477)


## Future Aspects

1. **Browser Extension**: Develop an extension to detect deepfake content directly in the browser and provide user feedback.

2. **Social Media Links Support**: Integrate functionality to analyze and detect deepfakes in videos shared via social media links.

3. **Detection Using Audio Anomalies**: Add audio analysis to identify anomalies that may indicate deepfake content.

4. **Frame/Frames Responsible for DeepFake Flag**: Implement a feature to identify specific frames responsible for deepfake detection, enhancing the granularity of analysis.


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
### Tech Stack Used

**WEB DEVELOPMENT**

- **Streamlit:** For building interactive web interfaces for the deepfake detection system and blockchain integration.
- **Web3:** To interact with the Ethereum blockchain for querying and updating smart contracts.

**BACKEND**

- **FastAPI:** For handling video uploads, deepfake detection, and API responses efficiently.



## Team:
Dhruv Desai<br/>
Atharva Humane<br/>
Niranjan More<br/>
Mithilesh Singh<br/>
Vaishnavi Hud<br/>
Kasturi Pawar<br/>
