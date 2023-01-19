"""
Copyright: Donghu Guo

Author: Donghu Guo

Description: this is the python script to extract sensor data from CFD simulations according to sensor coordinates

Github Repository: https://github.com/ese-msc-2021/irp-dg321
"""
import os
import sys
sys.path.append("..")
import tools
import vtk
import numpy as np


for i in range(1, 9):
    print('implementing case{}'.format(i))
# ###################################################################################
    # *************values setting***********
    root_path = '..data/' # set the root path where data is stored
    path = root_path + 'Cotrace_fixed_720_cases/case{}/'.format(i)   # The path where the vtu files are located
    probe_data_path = root_path + 'Cotrace_fixed_720_cases/case{}_npys/case{}_sensor.npy'.format(i, i)
    times = 720
    name_simu = 'Cotrace_fixed'
    vtu_start = 1
    vtu_end = 721  # 721
    vtu_step = 1
    fields_list = ['Tracer', 'Velocity', 'Temperature', 'Humidity', 'Virus1']

    x = [10.3, 6.8, 3.1]
    y = [5.2, 0.8, 5.0]
    z = [0.4, 1.0, 1.5, 2.25, 3.0, 3.5]

    # coordinates = [(5.8,6.1,0.8), (6.4,6.1,0.8)]
    # *************values setting***********


    # --------------------------#
    # -- Coordinates Fluidity --#
    # --------------------------#
    coordinates = []

    for i in range(6):
        coord1 = (x[0], y[0], z[i])
        coordinates.append(coord1)
        coord2 = (x[1], y[1], z[i])
        coordinates.append(coord2)
        coord3 = (x[2], y[2], z[i])
        coordinates.append(coord3)

    coordinates = np.array(coordinates)


    # ------------------------------------------------#
    # - Function to initialise vtk files             -#
    # ------------------------------------------------#
    def Initialisation(filename):
        '''
        This function initialises the vtk file
        Parameters
        ----------

        filename : string
            The filename of the vtk file

        Returns
        ---------

        ugrid

        '''
        # Read file
        if filename[-4:] == ".vtu":
            gridreader = vtk.vtkXMLUnstructuredGridReader()
        elif filename[-5:] == ".pvtu":
            gridreader = vtk.vtkXMLPUnstructuredGridReader()
        gridreader.SetFileName(filename)
        gridreader.Update()
        ugrid = gridreader.GetOutput()

        return ugrid

    # ------------------------------------------------#
    # - Function to initialise probe filter          -#
    # ------------------------------------------------#


    def InitialisePointData(ugrid, coordinates):

        # Initialise probe
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()

        # Create points to be extracted
        NrbPoints = 0
        for nodeID in range(len(coordinates)):
            NrbPoints += 1
            points.InsertNextPoint(
                coordinates[nodeID][0],
                coordinates[nodeID][1],
                coordinates[nodeID][2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        probe = vtk.vtkProbeFilter()

        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            probe.SetInput(polydata)
            probe.SetSource(ugrid)
        else:
            probe.SetInputData(polydata)
            probe.SetSourceData(ugrid)

        probe.Update()

        return probe, points, NrbPoints
    # ------------------------------------------------#
    # - Function to initialise cell filter           -#
    # ------------------------------------------------#


    def InitialiseCellLocator(ugrid):

        # Initialise locator
        CellLocator = vtk.vtkCellLocator()
        CellLocator.SetDataSet(ugrid)
        CellLocator.Update()

        return CellLocator


    # ---------------------------------------------------------------------
    # EXTRACT DATA
    # ---------------------------------------------------------------------
    def load_coord_data(filename, fields_list):

        # - Fields
        FIELDS = []

        for f in range(len(fields_list)):
            # - Field
            CO2 = []
            # - Time
            TimeVTU = []
            r = 0
            CO2.append([])
            fieldname = fields_list[f]
            # Read file
            ugrid = Initialisation(filename)
            # Initialise probe
            probe, points, NrbPoints = InitialisePointData(ugrid, coordinates)
            # Initialise cell location
            CellLocator = InitialiseCellLocator(ugrid)
            # -- Check Validity of points
            valid_ids = probe.GetOutput().GetPointData().GetArray('vtkValidPointMask')
            validPoints = tools.arr([valid_ids.GetTuple1(i)
                                    for i in range(NrbPoints)])
            for nodeID in range(len(coordinates)):
                # If valid point, extract using probe,
                # Otherwise extract the cell:
                #    If no cell associated - then it is really a non-valid point outside the domain
                #    Otherwise: do the average over the cell values - this provide the tracer value.
                # We need to do that as it is a well-known bug in vtk libraries -
                # sometimes it returns an invalid node while it is not...
                if validPoints[nodeID] == 1:
                    tmp = probe.GetOutput().GetPointData().GetArray(fieldname).GetTuple(nodeID)
                    CO2[r].append(tmp)
                else:
                    coord_tmp = np.array(points.GetPoint(nodeID))
                    # cell ID which contains the node
                    cellID = CellLocator.FindCell(coord_tmp)
                    idlist = vtk.vtkIdList()
                    ugrid.GetCellPoints(cellID, idlist)
                    pointsID_to_cellID = np.array([idlist.GetId(k) for k in range(
                        idlist.GetNumberOfIds())])  # give all the points asociated with this cell
                    # Non-valid points - We assign negative value - like that we
                    # know we are outside the domain
                    if len(pointsID_to_cellID) == 0:
                        CO2[r].append(-1e20)
                    else:
                        tmp = 0
                        for pointID in pointsID_to_cellID:
                            tmp += ugrid.GetPointData().GetArray(
                                fieldname).GetTuple(pointID)[0]
                        tmp = tmp / len(pointsID_to_cellID)
                        CO2[r].append(tmp)
                # print("nodeID: ", valid_ids)

            # Time
            time_tmp = probe.GetOutput().GetPointData().GetArray('Time').GetValue(0)
            TimeVTU.append(time_tmp)

            r += 1

            CO2 = np.array(CO2)

            FIELDS.append(CO2)

        FIELDS = np.concatenate(FIELDS, axis=2)

        return FIELDS


    results = []
    for t in range(vtu_start, vtu_end, vtu_step):
        filename = path + 'Cotrace_fixed_{}'.format(t) + '.vtu'
        result = load_coord_data(filename, fields_list)
        results.append(result)

    results_array = np.concatenate(results, axis=0)
    print(results_array.shape)

    np.save(probe_data_path , results_array)
