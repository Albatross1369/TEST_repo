import click    # package for creating command line interfaces
import numpy as np   #used to deal with array and matrices
import pandas as pd  #for data manipulation and analysis
import logging  #module for logging messages in Python
from rich.logging import RichHandler
from tqdm import tqdm
import sys
import yaml
import os
import pyvista as pv
import calc_and_mig_kx_ky_kz
from typing import Union
from mpi4py import MPI
import ctypes

comm = MPI.COMM_WORLD  #we initialize 3 MPI (Message Passing Interface) communication varibles here
rank = comm.Get_rank() #rank of current process in communicator
size = comm.Get_size() #total number of process in the communicator

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format="Rank: " + str(rank) + "/" + str(size) + ": %(asctime)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("rich")

#The above line initializes a logger called log.
#Doing this helps us debug and troubleshoot our code later



class Mesh:
    def __init__(self, vtk_file_name: Union[str, os.PathLike]): # Mesh object is initialzied, this function takes a VTK(Visualization Toolkit) file as input
        log.debug("Reading vtk file %s" % vtk_file_name) #logs a debug message
        
        if os.path.exists(vtk_file_name):       #here we are checking if the VTK file exists or not
            self.vtk_file_name = vtk_file_name  #if a VTK file is present we assign the self object
        else:
            msg = "VTK file %s does not exist" % vtk_file_name # we get error message if there is no VTK file
            log.error(msg)
            raise ValueError(msg)
        try:                                # we use try block to read VTK file using read function
            self.mesh = pv.read(self.vtk_file_name) 
        except Exception as e:          # if we get exception, message is logged and error is raised
            log.error(e)
            raise ValueError(e)
        log.debug("Reading mesh completed") #mesh reading completed

        # now we set the Mesh objects attributes
        self.npts = self.mesh.n_points  #number of nodes in the mesh
        self.ncells = self.mesh.n_cells #number of cells in the mesh
        self.nodes = np.array(self.mesh.points) #array of mesh nodes coodinates
        self.tet_nodes = self.mesh.cell_connectivity.reshape((-1, 4)) #arrays that makes up each tetrahedral cell in the mesh
        log.debug("Generated mesh properties")

    def __str__(self):
        return str(self.mesh) #string representation of the object

    def get_centroids(self): # we are going to calculate the centroids of the tetrahedral cells in the mesh
        log.debug("Getting centroids")
        nk = self.tet_nodes[:, 0]     #These are the 4 vertices of the tertrahedron
        nl = self.tet_nodes[:, 1]
        nm = self.tet_nodes[:, 2]
        nn = self.tet_nodes[:, 3]
        self.centroids = (            #centroids are the average of the coordinates of the four vertics
            self.nodes[nk, :]
            + self.nodes[nl, :]
            + self.nodes[nm, :]
            + self.nodes[nn, :]
        ) / 4.0
        log.debug("Getting centroids done!") #calc of centroids complete

    def get_volumes(self):
        log.debug("Getting volumes")
        ntt = len(self.tet_nodes)       
        vot = np.zeros((ntt))       #creating a zero filled array (maybe to fill in the volumes of tertrahedron)
        for itet in np.arange(0, ntt):  # this loop iterates all over the tetrahedra and extracts the node indices and coordinates
            n1 = self.tet_nodes[itet, 0]
            n2 = self.tet_nodes[itet, 1]
            n3 = self.tet_nodes[itet, 2]
            n4 = self.tet_nodes[itet, 3]
            x1 = self.nodes[n1, 0]
            y1 = self.nodes[n1, 1]
            z1 = self.nodes[n1, 2]
            x2 = self.nodes[n2, 0]
            y2 = self.nodes[n2, 1]
            z2 = self.nodes[n2, 2]
            x3 = self.nodes[n3, 0]
            y3 = self.nodes[n3, 1]
            z3 = self.nodes[n3, 2]
            x4 = self.nodes[n4, 0]
            y4 = self.nodes[n4, 1]
            z4 = self.nodes[n4, 2]
            pv = (                  #vol of tetrahedron is calculates here
                (x4 - x1) * ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
                + (y4 - y1) * ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
                + (z4 - z1) * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            )
            vot[itet] = np.abs(pv / 6.0) # absolute value of the result is stored here
        self.volumes = vot #storing this as an attribute of mesh obj
        log.debug("Getting volumes done!") #completed getting volumes




class MagneticProperties:
    def __init__(              #magnetic susceptability is the measure of how much a material will be magnetized in an applied magnetic field.
        self,
        file_name: Union[str, os.PathLike],
        kx: float = 1.0,      #magnetic susceptability in x-direction
        ky: float = 1.0,      #magnetic susceptability in y-direction
        kz: float = 1.0,      #magnetic susceptability in z-direction
    ):
        log.debug("Reading magnetic properties %s" % file_name) #logs the file
        if os.path.exists(file_name): #checks if file exists 
            self.file_name = file_name 
        else:
            msg = "File %s does not exist" % file_name #if file does not exist we get an error
            log.error(msg)
            raise ValueError(msg)

        try:
            self.properties = np.load(file_name) #loads magnetic properties
        except Exception as e:
            log.error(e)
            raise ValueError(e)     #we get get ValueError if there is any error
        log.debug("Reading magnetic properties %s done!" % file_name) #reading mag properties done

        if len(self.properties.shape) > 0: #checks if shape of the properties is non-empty
            self.n_cells = self.properties.shape[0] #if so, it is set to number of cells in the array
        else:
            msg = "Magnetic properties file %s is incorrect" % file_name
            log.error(msg)
            raise ValueError(msg)   #if properties array is empy the get Value error

        if self.properties.ndim == 1: #check if magnetic properties are 1D
            self.properties = np.expand_dims(self.properties, axis=1) # if so it is expanded to 2D

        if self.properties.shape[1] > 0: #checks if it has at least one colomn
            self.susceptibility = self.properties[:, 0] # if so, it sets susceptability to that colomn

        if self.properties.shape[1] > 1: #checks if it has at least 2 colomns
            self.kx = self.properties[:, 1] #if yes, sets kx to 2nd colomn
        else:
            self.kx = np.full((self.n_cells,), kx)

        if self.properties.shape[1] > 2: #checks if the is has at least 3 colomns
            self.ky = self.properties[:, 2] #if yes, sets ky to 3rd colomn
        else:
            self.ky = np.full((self.n_cells,), ky)

        if self.properties.shape[1] > 3: #checks if the is has at least 4 colomns
            self.kz = self.properties[:, 3] #if yes, sets ky to 4th colomn
        else:
            self.kz = np.full((self.n_cells,), kz)

        log.debug("Setting all magnetic properties done!") #completed setting magnetic properties



class MagneticAdjointSolver:
    def __init__(
        self,
        reciever_file_name: Union[str, os.PathLike],
        Bx: float = 4594.8, #magnetic field strength in x direction 
        By: float = 19887.1, #magnetic field strength in y direction
        Bz: float = 41568.2, #magnetic field strength in z direction
    ):
        log.debug("Solver initialization started!")
        if os.path.exists(reciever_file_name): # checks if file path exists 
            self.reciever_file_name = reciever_file_name
        else:
            msg = "File %s does not exist" % reciever_file_name
            log.error(msg)
            raise ValueError(msg) #else we get a Value Error

        try:
            self.receiver_locations = pd.read_csv(reciever_file_name) #we read the csv file using pandas
        except Exception as e:
            log.error(e)
            raise ValueError(e) #if any error, then a Value error is raised

        self.Bx = Bx # we set the instance variables
        self.By = By
        self.Bz = Bz
        self.Bv = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2) #calculating magnitude using pythagoras theorm
        self.LX = np.float32(self.Bx / self.Bv) # we calculate unit vectors by dividing magnetic feild by magnitude
        self.LY = np.float32(self.By / self.Bv)
        self.LZ = np.float32(self.Bz / self.Bv)
        log.debug("Solver initialization done!")

    def solve(self, mesh: Mesh, magnetic_properties: MagneticProperties): # takes mesh obj and magnetic properties as input
        log.debug("Solver started for %s" % magnetic_properties.file_name)
        rho_sus = np.zeros((10000000), dtype="float32") #here we are going to create several arrays
        rho_sus[0 : mesh.ncells] = magnetic_properties.susceptibility # will store the magnetic susceptibility values.

        KXt = np.zeros((10000000), dtype="float32")
        KXt[0 : mesh.ncells] = magnetic_properties.kx #will store the kx values for each cell in the mesh.

        KYt = np.zeros((10000000), dtype="float32")
        KYt[0 : mesh.ncells] = magnetic_properties.ky #will store the ky values for each cell in the mesh.

        KZt = np.zeros((10000000), dtype="float32")
        KZt[0 : mesh.ncells] = magnetic_properties.kz #will store the kz values for each cell in the mesh.

        ctet = np.zeros((10000000, 3), dtype="float32")
        ctet[0 : mesh.ncells] = np.float32(mesh.centroids) #will store the centroid coordinates for each cell in the mesh.

        vtet = np.zeros((10000000), dtype="float32")
        vtet[0 : mesh.ncells] = np.float32(mesh.volumes) #which will store the volume of each cell in the mesh.

        nodes = np.zeros((10000000, 3), dtype="float32")
        nodes[0 : mesh.npts] = np.float32(mesh.nodes) #will store the coordinates of all nodes in the mesh.

        tets = np.zeros((10000000, 4), dtype=int)
        tets[0 : mesh.ncells] = mesh.tet_nodes + 1 #will store the indices of nodes that make up each tetrahedron in the mesh.

        n_obs = len(self.receiver_locations) #sets number of observations as length of reciver locations
        rx_loc = self.receiver_locations.to_numpy() #converts reciever locations into numpy array

        obs_pts = np.zeros((1000000, 3), dtype="float32")#initializes a numpy array to store the observation points.
        obs_pts[0:n_obs] = np.float32(rx_loc[:, 0:3])

        ismag = True
        rho_sus = rho_sus * self.Bv #here we multiply the susceptibility values by the magnitude of magnetic field

        istensor = False

        mig_data = calc_and_mig_kx_ky_kz.calc_and_mig_field(
            rho_sus,
            ismag,          #this calls a function calc_and_mig_field
            istensor,       #We input all the arrays required
            KXt,           
            KYt,            
            KZt,
            self.LX,
            self.LY,
            self.LZ,
            nodes,
            tets,
            mesh.ncells,
            obs_pts,
            n_obs,
            ctet,
            vtet,
        )
        log.debug("Solver done for %s" % magnetic_properties.file_name)
        return mig_data[0 : mesh.ncells] #returns calculated magnetic data


@click.command() #referencing command line interface
@click.option(
    "--config_file",    #specifying a configuration file
    help="Configuration file in YAML format",
    type=click.Path(),
    required=True,
    show_default=True,
)


def isciml(config_file: os.PathLike): #path to the configeration file
    log.debug("Reading configuration file") #loggin the debug message
    if os.path.exists(config_file): #checks if specified file exist
        try:
            with open(config_file, "r") as fp: #opens and reads file using yaml.safe_load 
                config = yaml.safe_load(fp)
        except Exception as e:
            log.error(e)
            raise ValueError(e) # we get Value error if any errors during reading the file
    else:
        msg = "File %s doesn't exist" % config_file
        log.error(msg)
        raise ValueError(msg) # we get Value error if specified config does not exist

    log.debug("Reading configuration file done!")

    mesh = Mesh(config["vtk_file"]) #mesh obj created using VTK file config
    mesh.get_centroids() #we call this method to compute centriods
    mesh.get_volumes() #we call this methods to compute volumes

    properties = MagneticProperties(config["magnetic_properties_file"]) #we create MagneticProperties object
    solver = MagneticAdjointSolver(config["receiver_locations_file"]) #we create MagneticAdjointSolver object
    output = solver.solve(mesh, properties) # upon solving we assign output data to output variable
    # np.save("output.npy", output)
    return 0 # the function then resturns 0


if __name__ == "__main__":
    sys.exit(isciml())  # pragma: no cover
