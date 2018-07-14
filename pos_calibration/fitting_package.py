import cv2, math
import matplotlib.pyplot as plt
import numpy as np
import os, pickle

class fittingPackage():
    def __init__(self, line_top, square_vertex, border_vertex):
            self.line_top = line_top
            self.square_vertex = square_vertex
            self.border_vertex = border_vertex

    def __getCoord(self, img):
        fig = plt.figure()
        # img = cv2.flip(img, 1)  # We horizontally flip the image
        plt.imshow(img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        cid2 = fig.canvas.mpl_connect('scroll_event', self.__onscroll__)
        plt.show()
        return (self.point[1], self.point[0])  # Fix axis values

    def __onclick__(self, click):
        self.point = (click.xdata, click.ydata)
        plt.close('all')
        return self.point

    def __onscroll__(self, click):
        self.point = (-1, -1)
        plt.close('all')
        return self.point

    def __get_contact_info(self, directory, num):
        def get_cart(path):
            cart = np.load(path)
            a = cart[3]
            cart[3] = cart[6]
            b = cart[4]
            cart[4] = a
            c = cart[5]
            cart[5] = b
            cart[6] = c
            return cart

        def load_obj(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        directory += '/p_' + str(num)
        file_list = os.listdir(directory)

        cart = get_cart(directory + '/cart.npy')
        gs1_list = []
        gs2_list = []
        wsg_list = []

        for elem in file_list:
            path = directory + '/' + elem
            if 'GS1' in elem:
                gs1_list.append(cv2.imread(path))
            elif 'GS2' in elem:
                gs2_list.append(cv2.imread(path))
            elif 'wsg' in elem:
                wsg_list.append(load_obj(path))

        return cart, gs1_list, gs2_list, wsg_list

    def get_real_point_list(self, name):
        if name == "squares":
            squares_distance = [
                (.0, .0),
                (.0, 7.3),
                (.0, 18.5),
                (.0, 25.38),

                (11.12, 0),
                (11.12, 7.3),
                (11.12, 18.5),
                (11.12, 25.38),

                (20.02, 0),
                (20.02, 7.3),
                (20.02, 18.5),
                (20.02, 25.38),

                (30.4, 0),
                (30.4, 7.3),
                (30.4, 18.5),
                (30.4, 25.38)
            ]
            point_list = []
            for elem in squares_distance:
                x = float(self.square_vertex[0] + elem[1])
                y = float(self.square_vertex[1])
                z = float(self.square_vertex[2] - elem[0]) # - (250+72.5+139.8)  # We substract the mesuring tool
                point_list.append((x, y, z))

        elif name == "line":
            distances = [
                .0,
                33.
            ]
            point_list = []
            for dist in distances:
                x = float(self.line_top[0])
                y = float(self.line_top[1])
                z = float(self.line_top[2] - dist) # - (250+72.5+139.8)  # We substract the mesuring tool
                point_list.append((x, y, z))

        elif name == "border":
            squares_distance = [
                (.0, -15.),
                (.0, 0.),
                (15., 0.),

                (15.+8.3, 0.),
                (15.+8.3+15., 0.),
                (15.+8.3+15., -15.)
            ]
            point_list = []
            for elem in squares_distance:
                x = float(self.border_vertex[0])
                y = float(self.border_vertex[1] + elem[1])
                z = float(self.border_vertex[2] - elem[0]) # - (250+72.5+139.8)  # We substract the mesuring tool
                point_list.append((x, y, z))

        return point_list

    def create_package(self, name, gs_id, from_list=False, from_path='', touches_list=[0], save_path='', previous=''):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        prev = None
        try:
            prev = np.load(previous).tolist()
        except Exception as e:
            print e

        real_point_list = self.get_real_point_list(name=name)
        mc = []
        for i in touches_list:
            # We load the information about that touch
            cart, gs1_list, gs2_list, wsg_list = self.__get_contact_info(from_path, i)

            # We reshape the gripper state information
            gripper_state = {}
            gripper_state['pos'] = cart[0:3]
            gripper_state['quaternion'] = cart[-4:]
            gripper_state['Dx'] = wsg_list[0]['width']/2.0
            gripper_state['Dz'] = 139.8 + 72.5 + 160  # Base + wsg + finger

            for j in range(len(real_point_list)):
                print "Touch point: " + str(j)
                if (prev is not None) and (len(prev)>(i*len(real_point_list) + j)):
                    mc.append(prev[i*len(real_point_list) + j])
                    print "[Loaded from memory] " + str(len(prev))
                else:
                    if gs_id == 1:
                        gs_point = self.__getCoord(gs1_list[0])
                    elif gs_id == 2:
                        gs_point = self.__getCoord(gs2_list[0])

                    if gs_point != (-1, -1):
                        real_point = real_point_list[j]
                        print "Matched: " + str(gs_point) + " with " + str(real_point)
                        mc.append((gs_point, gs_id, gripper_state, real_point))
                    else:
                        print "Point cancelled"
            # We save at each contact
            np.save(save_path + '/' + name + '.npy', mc)
            print "Saved"


if __name__ == "__main__":
    name = 'squares'
    from_path = '/media/mcube/data/shapes_data/pos_calib/squares'
    touches_list = range(24)
    save_path = '/media/mcube/data/shapes_data/pos_calib/marked'
    already_done='/media/mcube/data/shapes_data/pos_calib/marked/squares.npy'
    gs_id = 2
    #
    # name = 'border'
    # from_path = '/media/mcube/data/shapes_data/pos_calib/border'
    # touches_list = range(29)
    # save_path = '/media/mcube/data/shapes_data/pos_calib/marked'
    # already_done=''
    # gs_id = 2

    # name = 'line'
    # from_path = '/media/mcube/data/shapes_data/pos_calib/line'
    # touches_list = range(33)
    # save_path = '/media/mcube/data/shapes_data/pos_calib/marked'
    # already_done=''
    # gs_id = 2

    # already_done = ''
    fp = fittingPackage(
        line_top=(1299, 253, 463),
        square_vertex=(476, 605, 386),
        border_vertex=(696, 663, 383)
    )
    fp.create_package(
        name=name,
        gs_id=gs_id,
        from_list=False,
        from_path=from_path,
        touches_list=touches_list,
        save_path=save_path,
        previous=already_done
    )
