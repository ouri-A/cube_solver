# One scan that read a legal cube: what the cube is, how it looks, and how to solve it.

std_msgs/Header header          # stamp: when the scan finished

string state                    # 54 face letters in kociemba order, faces U R F D L B
std_msgs/ColorRGBA[6] colours   # each face letter's colour as the camera read it, U R F D L B, 0-1
string[] solution               # kociemba's moves in order, e.g. ["R", "U2", "F'"]; empty if solved
