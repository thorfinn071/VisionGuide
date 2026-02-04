# VisionGuide 

**VisionGuide** is an assistive computer‑vision application designed to help blind and visually impaired people navigate their surroundings more safely. The system uses a camera, real‑time object detection, distance estimation, and voice feedback to describe the environment and warn about nearby obstacles.

The project focuses on **simplicity, stability, and real‑time performance**, making it suitable for continuous everyday use.

---

##  Key Features

###  Real‑Time Object Detection (YOLO)

* Detection of people and common outdoor objects
* Extended *street mode* with relevant classes (person, door, tree, stairs, traffic light, etc.)
* Confidence and bounding‑box size filtering to reduce noise

###  Distance Estimation (Bounding Box–Based)

* Distance is estimated using the **relative size of bounding boxes**
* No depth models are used
* Stable FPS and predictable behavior
* Suitable for real‑time navigation scenarios

###  Voice Feedback

* Text‑to‑speech notifications for detected objects
* Distance‑aware warnings:

  * *Close*
  * *Very close*
* Objects in the **very close** zone are always announced
* Object position is included when relevant: *left / center / right*

###  Scan Mode

* Dedicated mode for describing the environment
* Announces all detected objects in the field of view in sequence
* Example output:

  > “In view: person, tree, car”
* Can be triggered on demand to quickly understand surroundings

###  Cane Mode

* Simplified mode focused on obstacles directly ahead
* Reduced number of voice notifications
* Designed for continuous use while walking

---

##  Technology Stack

* **Python 3.10+**
* **Ultralytics YOLO**
* **OpenCV**
* **NumPy**
* **PyTorch**
* **SpeechRecognition**
* **Windows TTS (SAPI)**

---

##  Project Structure (Simplified)

```
VisionGuide/
│
├── main.py                # Main application entry point
├── models/                # YOLO models (custom / pretrained)
├── utils/                 # Helper functions
├── assets/                # Resources
├── requirements.txt
└── README.md
```

---

##  Running the Project

1. Clone the repository:

```bash
git clone https://github.com/thorfinn071/VisionGuide.git
cd VisionGuide
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python main.py
```

---

##  Social Impact

VisionGuide aims to:

* Improve independent mobility for blind and visually impaired users
* Reduce the risk of collisions and accidents
* Make modern AI‑based assistive technology more accessible

The project can be further extended to mobile platforms and wearable devices.

---

##  Authors

Developed by a student team as an AI project focused on accessibility and real‑world impact.

**Contributors:**

* Nurmukhamed Sabit
* Alizhan Myrazabek

---

⭐ If you find this project useful, consider giving it a star on GitHub!
