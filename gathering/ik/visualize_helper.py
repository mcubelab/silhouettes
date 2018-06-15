import os, errno
import numpy as np

from visualization_msgs.msg import MarkerArray
from marker_helper import createCubeMarker2, createDeleteAllMarker
from roshelper import pubFrame
#from helper import reference_frames, get_params_yaml, quat_from_matrix
import helper
import matplotlib.cm

def visualize_grasping_proposals(proposal_viz_array_pub, proposals, listener, br, is_selected = False):

    n = proposals.shape[0]
    if n>0 and proposals[0] is None:
        return

    scale = (0.001,0.02,0.2)
    markers_msg = MarkerArray()
    #if not is_selected:  # don't delete other candidate
    #    markers_msg.markers.append(createDeleteAllMarker('pick_proposals'))
    
    proposals_score = proposals[0:min(n, 10), 11]
    if len(proposals_score) > 0:
        max_score = np.amax(proposals_score)
        min_score = np.amin(proposals_score)   
        print max_score, min_score 
    for i in range(0,min(n, 10)):
        pick_proposal = proposals[i,:]
        objInput = pick_proposal
        
         #~get grasp pose and gripper opening from vision
        if len(objInput)==12:
            graspPos, hand_X, hand_Y, hand_Z, grasp_width = helper.get_picking_params_from_12(objInput)
            graspPos = graspPos #+ hand_X*0.02*1
            grasp_score = objInput[-1]
        elif len(objInput)==7:
            graspPos, hand_X, hand_Y, hand_Z, grasp_width = helper.get_picking_params_from_7(objInput, 'dummy', listener, br)
            graspPos = graspPos #+ hand_X*0.02*1
            grasp_score = 1
        
        color = matplotlib.cm.seismic((1-(grasp_score-min_score)/(max_score-min_score)))
        if is_selected:
            color = (0, 1, 0, 1)
        
        scale_bar = (grasp_width,0.003,0.2)
        #import ipdb; ipdb.set_trace()
        rotmat = np.vstack((hand_X, hand_Y, hand_Z, np.zeros((1,3)))).T
        rotmat = np.vstack((rotmat, np.array([[0,0,0,1]])))
        quat = helper.quat_from_matrix(rotmat)
        

        m1 = createCubeMarker2(rgba = color, scale = scale, offset = tuple(graspPos+hand_X*grasp_width/2), orientation= tuple(quat), marker_id = i*3, ns = 'pick_proposals')
        m2 = createCubeMarker2(rgba = color, scale = scale, offset = tuple(graspPos-hand_X*grasp_width/2), orientation= tuple(quat), marker_id = i*3+1, ns = 'pick_proposals')
        m3 = createCubeMarker2(rgba = color, scale = scale_bar, offset = tuple(graspPos), orientation= tuple(quat), marker_id = i*3+2, ns = 'pick_proposals')

        markers_msg.markers.append(m1)
        markers_msg.markers.append(m2)
        markers_msg.markers.append(m3)
        
    for i in range(0,10):
        proposal_viz_array_pub.publish(markers_msg)
    #pauseFunc(True)
