VisionGuide

VisionGuide is a computer program that helps people who're blind or have trouble seeing. It uses a camera to look and figure out what is nearby. The VisionGuide system can tell what things are around it and how away they are. Then it uses a voice to tell the person about what it sees so they can avoid bumping into things. The VisionGuide system is really good, at helping people navigate their surroundings safely.

The project is about keeping things simple and stable. It also works well in real time. This makes the project something that people can use every day without any problems. The project is good, for real time performance. That is what people need for everyday use.

Key Features

Real Time Object Detection (YOLO)

• Detection of people and common outdoor objects

• Extended street mode with relevant classes (person, door, tree, stairs, traffic light, etc.)

• Confidence and bounding box size filtering to reduce noise

Distance Estimation (Bounding Box–Based)

• The distance is figured out by looking at how big the boxesre in relation to each other. We use the size of these boxes to estimate the distance. The boxes are like frames around objects. We look at how big they appear to be and that helps us estimate the distance of the objects, from us using the relative size of these boxes.

• Stable FPS and predictable behavior

• Suitable for real time navigation scenarios

Voice Feedback

• Text to speech notifications for detected objects

• Distance aware warnings:

o Close

o Very close

• Things that are really close by will always be told to you the objects that are in this close zone will always be announced, so you know what the objects, in the very close zone are.

• The position of the object is included when it is relevant and it can be on the left or in the center or, on the right.

Scan Mode (Environment Overview)

• Dedicated mode for describing the environment

• Announces all detected objects in the field of view in sequence

• Example output:

“In view: person, tree, car”

• The thing can be started when you want it to so it can quickly figure out what is around it. Understand the surroundings really fast, like the things that are near, to the thing and what the surroundings are.

Cane Mode

• The simplified mode is about the obstacles that are right in front of the game. It helps you deal with the things that're directly ahead of you. This mode is really good at showing you what is coming up next like the obstacles that you need to get. The simplified mode is great because it focuses on the obstacles that're right, in your path.

• Reduced number of voice notifications

• Designed for continuous use while walking

 Technology Stack

• Python 3.10+

• Ultralytics YOLO

• OpenCV

• NumPy

• PyTorch

• SpeechRecognition

• Windows TTS (SAPI)

Running the Project

1. Clone the repository:

git clone https://github.com/thorfinn071/VisionGuide.git

cd VisionGuide

2. Install dependencies:

pip install -r requirements.txt

3. Run the application:

python main.py

 Social Impact

VisionGuide wants to do a things. The main goal of VisionGuide is to help people. VisionGuide aims to make things easier for everyone who uses it. The people, at VisionGuide want to make sure that VisionGuide does what it is supposed to do. So VisionGuide aims to do lots of things for people who need it. VisionGuide is really trying to make a difference.

• We need to make it easier for people and those who cannot see very well to get around on their own. This will help blind and visually impaired users to be more independent when they are moving around. The goal is to improve mobility, for blind and visually impaired users so they can go wherever they want without needing help from others.

• We should try to make the roads safer by reducing the risk of collisions and accidents. This will help prevent people from getting hurt in car accidents and collisions. Reducing the risk of collisions and accidents is very important, for our safety.

• We should make modern intelligence based assistive technology more accessible to people who need it so that modern artificial intelligence based assistive technology can really help them in their daily lives. This is very important, for artificial intelligence based assistive technology.

The project can also be used on phones and devices that people wear. This means the project can be used on platforms and wearable devices.

Authors

Developed by a student team as an AI project focused on accessibility and real world impact.

Contributors:

• Nurmukhamed Sabit

• Alizhan Myrzabek

If you find this project useful, consider giving it a star on GitHub!

