import cv2
import numpy as np
import matplotlib
import pandas as pd
# Model
import torch


PATH = 'part_A'


# # Load the video stream
# video = cv2.VideoCapture(0)

# while(True):
#    # Capture each frame as an image
#    ret, frame = video.read()

#    # show the image on the screen
#    cv2.imshow('frame', frame)
     
#    # Stop the playback when pressing ‘q’
#    if cv2.waitKey(1) == ord('q'):
#        cv2.imwrite('filename.jpg', frame)
#        break

# # Release the video from memory
# video.release() 

# # Clean up
# cv2.destroyAllWindows()

# Take NAO's picture

#ZMQ = False
ZMQ = True

if ZMQ:
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:5555')

    r = socket.recv_json()

while r!= 'end':
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/yolov5/supp_boxes.pt') 
    #print(predictions)

    # Images
    imgs = PATH + '/object.png'  # image

    # Inference
    results = model(imgs)

    # Results
    results.print()
    # results.save()
    results.show()

    results.xyxy[0]  # img1 predictions (tensor)
    print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

    if results.pandas().xyxy[0].index.size == 0:
        got_object = False
        print(got_object)
        socket.send_json('False')        

    color = (0, 0, 255) # RED
    for k, row in results.pandas().xyxy[0].iterrows():
        print(row.name, row.confidence)
        if row.confidence >= 0.4:
            conf = row.confidence
            top_left_x = int(row.xmax)
            top_left_y = int(row.ymax)
            bottom_right_x = int(row.xmin)
            bottom_right_y = int(row.ymin)

            mid_x = (top_left_x - bottom_right_x) // 2 + bottom_right_x
            mid_y = (top_left_y - bottom_right_y) // 2 + bottom_right_y

            print(mid_x, mid_y)

            pt1 = (int(row.xmin), int(row.ymin))
            pt2 = (int(row.xmax), int(row.ymax))
            image = cv2.imread(imgs)
            outfile = "bbox.jpg"
            frame = image[int(row.ymin):int(row.ymax), int(row.xmin):int(row.xmax)]
            cv2.imwrite(outfile, frame)

            cv2.imwrite(outfile, frame)

            #Reading the image with opencv
            img = cv2.imread(outfile)
            y, x = img.shape[0:2]
            print(x, y)
            b, g, r = img[int(y/2), int(x/2)]
            b = int(b)
            g = int(g)
            r = int(r)

            #Reading csv file with pandas and giving names to each column
            index=["color","color_name","hex","R","G","B"]
            csv = pd.read_csv(PATH + '/colors.csv', names=index, header=None)
        

            #function to calculate minimum distance from all colors and get the most matching color
            def getColorName(R,G,B):
                minimum = 10000
                for i in range(len(csv)):
                    d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
                    if(d<=minimum):
                        minimum = d
                        cname = csv.loc[i,"color_name"]
                return cname

            cv2.imshow("image",img)
    
            #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
            cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

            #Creating text string to display( Color name and RGB values )
            text = getColorName(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
            recognised_colour = getColorName(r,g,b)
            
            print(recognised_colour)
    
            if cv2.waitKey(20) & 0xFF ==27:
                break
            
            cv2.destroyAllWindows()

            colour = recognised_colour
            if colour in ["Plum (Traditional)", "Fandango", "Boysenberry", "Royal Fuchsia", "Mulberry", "Persian Pink", "Hot Pink", "Brilliant Rose", "Pale Magenta", "Sky Magenta", \
                        "Deep Carmine", "Tuscan Red", "Rich Maroon", "Mountbatten Pink", "Big Dip O'Ruby", "Aurometalsaurus", "Medium Ruby", "Rose Vale", "Copper Rose", "China Rose", \
                        "Raspberry Rose", "Raspberry Pink", "Rose Pink", "Rich Brilliant Lavender", "Pansy Purple", "Pink Sherbet", \
                        "Amaranth","Brink Pink","Cerise","Dogwood Rose","French Rose","Fuchsia Rose","Folly","Raspberry","Razzmatazz","Raspberry Rose","Rose red","Telemagenta"]:
                colour = "RED"
                supp_index = 'supp_01'
            elif colour in ["Baby Blue Eyes", "Maya Blue", "Light Sky Blue", "Cornflower Blue", "United Nations Blue", "Dodger Blue", "Bleu De France", "Steel Blue", "Lapis Lazuli", "Honolulu Blue", \
                            "Blue (Ncs)", "Yale Blue", "Ucla Blue", "Indigo (Dye)", "Sapphire Blue", "Teal Blue", "Sky Blue", "Air Force Blue (Raf)", "Air Force Blue (Usaf)", "Air Superiority Blue", "Alice Blue", \
                            "Ball Blue", "Baby Blue", "Columbia Blue", "Light Blue", "Light Sky Blue", "Periwinkle", "Powder Blue (Web)", "Sky Blue"]:
                colour = "BLUE"
                supp_index = 'supp_02'
            elif colour in ["Peach", "Persian Orange", "Bronze", "Copper", "Antique Brass", "Tiger'S Eye", "Peru", "Golden Brown", "Golden Yellow", "Fluorescent Orange", "Fluorescent Yellow", \
                            "Aureolin", "Canary Yellow", "Golden Yellow", "Icterine", "Jonquil", "Maize", "Mustard", "Naples Yellow", "School Bus Yellow", "Stil De Grain Yellow", "Sunglow", "Yellow"]:
                colour = "ORANGE"
                supp_index = 'supp_03'
            elif colour in ["Platinum", "White", "Floral White", "Pale Aqua", "Lavender Gray", "Wild Blue Yonder", "Pastel Blue", "Cadet Gray", "Trolley Gray", "Taupe Gray", "Dark Gray", "Ash Gray",\
                             "Manatee", "Gray (X11 Gray)", "Pale Cornflower Blue", "Cool Gray", \
                            "Ghost White","Magnolia","Mint Cream","White","White Smoke","Snow"]:
                colour = "WHITE"
                supp_index = 'supp_04'
            elif colour in ["Sea Green", "Hooker'S Green", "Viridian", "Feldgrau", "Cal Poly Green", "Stormcloud", "Asparagus", \
                            "Bright Green","Dark Pastel Green","Green (Color Wheel) (X11 Green)","Harlequin","Kelly Green","Lime Green","Malachite","Neon Green","Lawn Green","Yellow-Green"]:
                colour = "GREEN"
                supp_index = 'supp_05'
            elif colour in ["Pistachio", "Fern Green", "Olive Drab (Web) (Olive Drab #3)", "Camouflage Green", "Dark Tan", "Android Green", "Apple Green", \
                            "Green-Yellow", "Chartreuse (Traditional)","Chartreuse (Web)","Lime (Color Wheel)","Spring Bud","Pale Spring Bud","Mint Green","Pistachio"]:
                colour = "MINT"
                supp_index = 'supp_06'
            elif colour in ["Majorelle Blue", "Onyx", "Arsenic", "Outer Space", "Payne'S Gray", "Old Lavender", "Ucla Blue", "Davy'S Gray", "Light Slate Gray", "Cadet", "Quartz", "Purple Taupe", \
                            "Slate Gray", "Pastel Purple", "Dark Slate Blue", "Medium Slate Blue", "Ube", "Royal Purple", \
                            "Amethyst","Blue-Violet","Dark Violet","Electric Purple","Iris","Lavender (Floral)","Royal Purple","Slate Blue","Veronica","Violet"]:
                colour = "PURPLE"
                supp_index = 'supp_07'
            else:
                colour = "NONE"
                supp_index = 'None'

            print(colour)
            print(supp_index)

            got_object = True
            print(got_object)
            socket.send_json(supp_index)
            break

        elif row.confidence < 0.40:
            got_object = False
            print(got_object)
            socket.send_json('False')
            break
        
    r = socket.recv_json()
socket.send(b'done')
socket.close()


