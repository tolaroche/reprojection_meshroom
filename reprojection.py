import cv2
import numpy as np
import sys
import glob
import os
import json
import openmesh

"""This is the numpy dtype that I will use to store the cameras intrinsic parameters describe is the .sfm file."""
intrinsicParamType = np.dtype({'names':['id', 'K', 'distCoef'],
                               'formats' : [np.int_, (np.float_, (3,3)), (np.float_, 5)]

})


"""This is the numpy dtype that I will use to store the cameras parameters (for a particular image) describe is the .sfm file."""
sfmCameraPoseType = np.dtype({'names':['idV', 'idP', 'path', 'name', 'R', 't', 'known', 'intrinsic'],
               'formats': [np.int_, np.int_, 'U150', 'U30', (np.float_, (3,3)), (np.float_, 3), np.bool_, intrinsicParamType]
})


def readSfmFile(path):
    """
        This function take the path of a .sfm file and return a numpy array containing all camera Poses contained in the .sfm file

        Parameters
        ----------

        path : path of the .sfm file

        Return
        ------
        cameraPoses : numpy array containing all the information from the .sfm file using the formalism of "sfmCameraPoseType" numpy dtype


    
    """
    
    f = open(path)

    data = json.load(f)

    nc = len(data['views'])
    ni = len(data['intrinsics'])


    cameraPoses = np.zeros(nc, dtype=sfmCameraPoseType)
    intrinsicParam = np.zeros(ni, dtype=intrinsicParamType)


    #We find the differents intrinsics parameters contained in the .sfm file
    for j,i in enumerate(data['intrinsics']):

        intrinsicParam[j]['id'] = i['intrinsicId']
        intrinsicParam[j]['K'][0][0] = float(i['pxFocalLength'])
        intrinsicParam[j]['K'][1][1] = float(i['pxFocalLength'])
        intrinsicParam[j]['K'][2][2] = 1
        intrinsicParam[j]['K'][0][2] = float(i['principalPoint'][0])
        intrinsicParam[j]['K'][1][2] = float(i['principalPoint'][1])

        for k in range(len(i['distortionParams'])):
            intrinsicParam[j]['distCoef'][k] = i['distortionParams'][k]
            

    #We fill the views information in our array
    for i, v in enumerate(data['views']):

        cameraPoses[i]['idV'] = v['viewId']
        cameraPoses[i]['idP'] = v['poseId']
        cameraPoses[i]['path'] = v['path']
        cameraPoses[i]['name'] = v['path'].split('/')[-1]

        intId = int(v['intrinsicId'])
        k = np.where(intrinsicParam['id'] == intId)[0][0]

        cameraPoses[i]['intrinsic'] = intrinsicParam[k]

    #for the images having poses information we fill the our array with the R and t matrices
    for v in data['poses']:

        id = int(v['poseId'])
        i = np.where(cameraPoses['idP'] == id)[0][0]


        j=0
        for k in range(3):
            for l in range(3):
                cameraPoses[i]['R'][l][k] = float(v['pose']['transform']['rotation'][j])
                j+=1

        c = np.empty(3)

        for j in range(3):
            c[j] = float(v['pose']['transform']['center'][j])

        cameraPoses[i]['t'] = -cameraPoses[i]['R']@c


    f.close()

    return cameraPoses




def projectPointsCloud(points, cameraPose):
    """
        This function take a 3D point cloud and a camera pose 
        and return the 2D points cooresponding after the reprojection
        and the distortion treatment.

        Parameters 
        ----------
        points : numpy array of shape (n,3)
        cameraPose : sfmCameraPoseType

        Return
        ------
        Pd : numpy array of shape (n,2), the points coordinate after reprojection and distortion treatment
        P : numpy array of shape (n,2), the point after reprojection but without distortion treatment
       
    """

    cx = cameraPose['intrinsic']['K'][0][2]
    cy = cameraPose['intrinsic']['K'][1][2]
    fx = cameraPose['intrinsic']['K'][0][0]
    fy = cameraPose['intrinsic']['K'][1][1]

    k1 = cameraPose['intrinsic']['distCoef'][0]
    k2 = cameraPose['intrinsic']['distCoef'][1]
    k3 = cameraPose['intrinsic']['distCoef'][2]

    #pass the points cloud in the coordinate system of the camera
    RPt = (cameraPose['R']@points.T).T + cameraPose['t']

    #we project the points
    Pp = RPt/RPt[:,2].reshape((-1,1))


    #Here we treat the distortion
    #we compute the distance of the projected points to the camera principal point
    r2 = Pp[:,0].T*Pp[:,0].T + Pp[:,1].T*Pp[:,1].T

    #we then compute the radial distortion factor
    gamma = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2

    #the tangential distortion term over the x axis 

    deltax = 2*cameraPose['intrinsic']['distCoef'][3]*Pp[:,0]*Pp[:,1] \
             + cameraPose['intrinsic']['distCoef'][4]*(r2 + 2*Pp[:,0])

    #and over the y axis 
    deltay = 2*cameraPose['intrinsic']['distCoef'][4]*Pp[:,0]*Pp[:,1] \
             + cameraPose['intrinsic']['distCoef'][3]*(r2 + 2*Pp[:,1])


    RPtd = np.empty(Pp.shape)
    RPtd[:,0] = gamma*RPt[:,0] + deltax
    RPtd[:,1] = gamma*RPt[:,1] + deltay
    RPtd[:,2] = RPt[:,2]

    #We project the points we the corrected distortion
    Pdp = RPtd/RPtd[:,2].reshape((-1,1))

    #We pass from the retinal coordinate system to the image coordinate system
    P = (cameraPose['intrinsic']['K']@Pp.T).T
    Pd = (cameraPose['intrinsic']['K']@Pdp.T).T

    return Pd[:,:2], P[:,:2]



def pointsCloudPlot(im, pc, color, radius = 0, alpha = 0.5):
    """
        This function take an image, a points cloud and plot 
        this points cloud on the image. The points are plot 
        with a specific color, size and with alpha blending 
    
        Parameters
        ----------
        im : numpy array of shape (n, m, 3), the image
        pc : numpy array of shape (n, 2), the points cloud
        color : numpy array of shape(3,), the color of the plotted points
        radius : int, the size of the points plotted
        alpha : float, the alpha blending value

        Return
        ------
        imf : numpy array of shape (n, m, 3), the image with the points plotted


    """


    imp = np.copy(im)

    for p in pc:
        if p[0]<im.shape[1] and p[0]>0 and p[1]<im.shape[0] and p[1]>0:
            imp = cv2.circle(imp, center = (int(p[0]), int(p[1])), radius = radius, color = color, thickness = -1)
        
    imf = alpha*im + (1-alpha)*imp

    return imf




if __name__ == "__main__":

    # Path of the .sfm file
    pathSfm = '/home/laroche/Desktop/reprojection/sfm2.sfm'
    
    #Path of the mesh to reproject (must be in the right coordinate system)
    pathMesh = '/home/laroche/Desktop/reprojection/icp_both/duct_scan_pc2.obj'

    # Path to the folder where we want to save the images with the reprojection
    pathSave = '/home/laroche/Desktop/reprojection/icp_both/reprojection_scan/'

    cameraPoses = readSfmFile(pathSfm)

    mesh = openmesh.read_trimesh(pathMesh)

    color = np.array([255,255,0])

    for i, cp in enumerate(cameraPoses):

        print(i)

        im = cv2.imread(cp['path'])
        imf = np.copy(im)

        if cp['known']:

            Pd, _ = projectPointsCloud(mesh.points(), cp)

            imf = pointsCloudPlot(imf, Pd, color = color, radius = 2, alpha = 0.5)
        
        cv2.imwrite(pathSave + cp['name'], imf)






