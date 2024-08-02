# state file generated using paraview version 5.11.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1421, 638]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView1.CenterOfRotation = [1.3350000381469727, 1.0000000134459697, 0.4000000059604645]
renderView1.HiddenLineRemoval = 1
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [2.2443703880956103, 3.8229609633679296, 6.328797808025983]
renderView1.CameraFocalPoint = [1.121496188565233, 0.9576472049711732, 0.46154362375589003]
renderView1.CameraViewUp = [-0.08185956073494072, 0.9016453090871404, -0.4246583908477941]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.417167106720281
renderView1.CameraParallelProjection = 1
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.Visibility = 1
renderView1.AxesGrid.XTitle = 'Streamwise'
renderView1.AxesGrid.YTitle = 'Wall-Normal'
renderView1.AxesGrid.ZTitle = 'Spanwise'
renderView1.AxesGrid.XTitleFontSize = 88
renderView1.AxesGrid.YTitleFontSize = 88
renderView1.AxesGrid.ZTitleFontSize = 88
renderView1.AxesGrid.GridColor = [0.4666666666666667, 0.4627450980392157, 0.4823529411764706]
renderView1.AxesGrid.ShowGrid = 1
renderView1.AxesGrid.AxesToLabel = 34
renderView1.AxesGrid.XLabelFontSize = 80
renderView1.AxesGrid.YLabelFontSize = 80
renderView1.AxesGrid.ZLabelFontSize = 80
renderView1.AxesGrid.XAxisUseCustomLabels = 1
renderView1.AxesGrid.XAxisLabels = [2.67, 0.6675, 2.0025, 1.335]
renderView1.AxesGrid.YAxisUseCustomLabels = 1
renderView1.AxesGrid.YAxisLabels = [0.0, 0.5, 1.5, 2.0, 0.25, 0.75, 1.25, 1.75, 1.0]
renderView1.AxesGrid.ZAxisUseCustomLabels = 1
renderView1.AxesGrid.ZAxisLabels = [0.0, 0.8, 0.2, 0.4, 0.6]
renderView1.AxesGrid.UseCustomBounds = 1
renderView1.AxesGrid.CustomBounds = [0.0, 2.67, 0.0, 2.0, 0.0, 0.8]

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1421, 638)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
eP_1_timestep_1258_62_Q_eventvtu = XMLUnstructuredGridReader(registrationName='EP_1_timestep_1258_62_Q_event.vtu', FileName=['/lscratch/pietero/andres_clone/DRL_POL/EP_1_timestep_1258_62_Q_event.vtu'])
eP_1_timestep_1258_62_Q_eventvtu.PointArrayStatus = ['Q']
eP_1_timestep_1258_62_Q_eventvtu.TimeArray = 'None'

# create a new 'Slice'
bottomplanexzy0 = Slice(registrationName='bottom plane (xz, y=0)', Input=eP_1_timestep_1258_62_Q_eventvtu)
bottomplanexzy0.SliceType = 'Plane'
bottomplanexzy0.HyperTreeGridSlicer = 'Plane'
bottomplanexzy0.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
bottomplanexzy0.SliceType.Origin = [1.3350000381469727, 1e-15, 0.4000000059604645]
bottomplanexzy0.SliceType.Normal = [0.0, 1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
bottomplanexzy0.HyperTreeGridSlicer.Origin = [1.3350000381469727, 1.0, 0.4000000059604645]

# create a new 'Slice'
sideplanezyx0 = Slice(registrationName='side plane (zy, x=0)', Input=eP_1_timestep_1258_62_Q_eventvtu)
sideplanezyx0.SliceType = 'Plane'
sideplanezyx0.HyperTreeGridSlicer = 'Plane'
sideplanezyx0.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
sideplanezyx0.SliceType.Origin = [1e-10, 1.0, 0.4000000059604645]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
sideplanezyx0.HyperTreeGridSlicer.Origin = [1.3350000381469727, 1.0, 0.4000000059604645]

# create a new 'Slice'
backplanexyz0 = Slice(registrationName='back plane (xy, z=0)', Input=eP_1_timestep_1258_62_Q_eventvtu)
backplanexyz0.SliceType = 'Plane'
backplanexyz0.HyperTreeGridSlicer = 'Plane'
backplanexyz0.Crinkleslice = 1
backplanexyz0.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
backplanexyz0.SliceType.Origin = [1.3350000381469727, 1.0000597845064476, 1e-11]
backplanexyz0.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
backplanexyz0.HyperTreeGridSlicer.Origin = [1.3350000381469727, 1.0000597845064476, 0.4000000059604645]

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=eP_1_timestep_1258_62_Q_eventvtu)
contour1.ContourBy = ['POINTS', 'Q']
contour1.ComputeGradients = 1
contour1.Isosurfaces = [0.5]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Smooth'
smooth1 = Smooth(registrationName='Smooth1', Input=contour1)
smooth1.NumberofIterations = 100

# create a new 'Clip'
invertedLocalVolume = Clip(registrationName='(inverted) Local Volume', Input=smooth1)
invertedLocalVolume.ClipType = 'Box'
invertedLocalVolume.HyperTreeGridClipper = 'Plane'
invertedLocalVolume.Scalars = ['POINTS', 'Q']
invertedLocalVolume.Invert = 0

# init the 'Box' selected for 'ClipType'
invertedLocalVolume.ClipType.Position = [1.335, 0.0, 0.4]
invertedLocalVolume.ClipType.Length = [1.335, 2.0, 0.4]

# init the 'Plane' selected for 'HyperTreeGridClipper'
invertedLocalVolume.HyperTreeGridClipper.Origin = [1.3350000381469727, 1.0000000134459697, 0.4000000059604645]

# create a new 'Clip'
witnessBoundingBox = Clip(registrationName='Witness Bounding Box', Input=smooth1)
witnessBoundingBox.ClipType = 'Box'
witnessBoundingBox.HyperTreeGridClipper = 'Plane'
witnessBoundingBox.Scalars = ['POINTS', 'Q']

# init the 'Box' selected for 'ClipType'
witnessBoundingBox.ClipType.Position = [0.0, 0.0, 0.4]
witnessBoundingBox.ClipType.Length = [1.335, 2.0, 0.4]

# init the 'Plane' selected for 'HyperTreeGridClipper'
witnessBoundingBox.HyperTreeGridClipper.Origin = [1.3350000381469727, 1.0000000134459697, 0.4000000059604645]

# create a new 'Outline'
witnessBoundingBoxOutline = Outline(registrationName='Witness Bounding Box Outline', Input=witnessBoundingBox)

# create a new 'Clip'
localVolume = Clip(registrationName='Local Volume', Input=smooth1)
localVolume.ClipType = 'Box'
localVolume.HyperTreeGridClipper = 'Plane'
localVolume.Scalars = ['POINTS', 'Q']

# init the 'Box' selected for 'ClipType'
localVolume.ClipType.Position = [1.335, 0.0, 0.4]
localVolume.ClipType.Length = [1.335, 2.0, 0.4]

# init the 'Plane' selected for 'HyperTreeGridClipper'
localVolume.HyperTreeGridClipper.Origin = [1.3350000381469727, 1.0000000134459697, 0.4000000059604645]

# create a new 'Outline'
localVolumeBoundingBox = Outline(registrationName='Local Volume Bounding Box', Input=localVolume)

# create a new 'CSV Reader'
filtered_witness_points_01csv = CSVReader(registrationName='filtered_witness_points_0-1.csv', FileName=['/lscratch/pietero/andres_clone/DRL_POL/filtered_witness_points_0-1.csv'])

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(registrationName='TableToPoints1', Input=filtered_witness_points_01csv)
tableToPoints1.XColumn = 'x'
tableToPoints1.YColumn = 'y'
tableToPoints1.ZColumn = 'z'

# create a new 'Plane'
backplanexyz0_1 = Plane(registrationName='back plane (xy, z=0) ')
backplanexyz0_1.Origin = [0.0, 0.0, 0.0]
backplanexyz0_1.Point1 = [0.0, 2.0, 0.0]
backplanexyz0_1.Point2 = [2.67, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from localVolume
localVolumeDisplay = Show(localVolume, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
localVolumeDisplay.Representation = 'Surface'
localVolumeDisplay.AmbientColor = [0.0, 0.0, 1.0]
localVolumeDisplay.ColorArrayName = ['POINTS', '']
localVolumeDisplay.DiffuseColor = [0.0, 0.0, 1.0]
localVolumeDisplay.SelectTCoordArray = 'None'
localVolumeDisplay.SelectNormalArray = 'Normals'
localVolumeDisplay.SelectTangentArray = 'None'
localVolumeDisplay.OSPRayScaleArray = 'Q'
localVolumeDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
localVolumeDisplay.SelectOrientationVectors = 'None'
localVolumeDisplay.ScaleFactor = 0.19982177942292767
localVolumeDisplay.SelectScaleArray = 'Q'
localVolumeDisplay.GlyphType = 'Arrow'
localVolumeDisplay.GlyphTableIndexArray = 'Q'
localVolumeDisplay.GaussianRadius = 0.009991088971146382
localVolumeDisplay.SetScaleArray = ['POINTS', 'Q']
localVolumeDisplay.ScaleTransferFunction = 'PiecewiseFunction'
localVolumeDisplay.OpacityArray = ['POINTS', 'Q']
localVolumeDisplay.OpacityTransferFunction = 'PiecewiseFunction'
localVolumeDisplay.DataAxesGrid = 'GridAxesRepresentation'
localVolumeDisplay.PolarAxes = 'PolarAxesRepresentation'
localVolumeDisplay.ScalarOpacityUnitDistance = 0.11980724997613167
localVolumeDisplay.OpacityArrayName = ['POINTS', 'Q']
localVolumeDisplay.SelectInputVectors = ['POINTS', 'Normals']
localVolumeDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
localVolumeDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
localVolumeDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from invertedLocalVolume
invertedLocalVolumeDisplay = Show(invertedLocalVolume, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
invertedLocalVolumeDisplay.Representation = 'Surface'
invertedLocalVolumeDisplay.AmbientColor = [0.611764705882353, 0.611764705882353, 0.611764705882353]
invertedLocalVolumeDisplay.ColorArrayName = ['POINTS', '']
invertedLocalVolumeDisplay.DiffuseColor = [0.611764705882353, 0.611764705882353, 0.611764705882353]
invertedLocalVolumeDisplay.Opacity = 0.35
invertedLocalVolumeDisplay.SelectTCoordArray = 'None'
invertedLocalVolumeDisplay.SelectNormalArray = 'Normals'
invertedLocalVolumeDisplay.SelectTangentArray = 'None'
invertedLocalVolumeDisplay.BackfaceAmbientColor = [0.5607843137254902, 0.5607843137254902, 0.5607843137254902]
invertedLocalVolumeDisplay.BackfaceDiffuseColor = [0.5098039215686274, 0.5098039215686274, 0.5098039215686274]
invertedLocalVolumeDisplay.BackfaceOpacity = 0.51
invertedLocalVolumeDisplay.OSPRayScaleArray = 'Q'
invertedLocalVolumeDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
invertedLocalVolumeDisplay.SelectOrientationVectors = 'None'
invertedLocalVolumeDisplay.ScaleFactor = 0.2670000076293945
invertedLocalVolumeDisplay.SelectScaleArray = 'Q'
invertedLocalVolumeDisplay.GlyphType = 'Arrow'
invertedLocalVolumeDisplay.GlyphTableIndexArray = 'Q'
invertedLocalVolumeDisplay.GaussianRadius = 0.013350000381469726
invertedLocalVolumeDisplay.SetScaleArray = ['POINTS', 'Q']
invertedLocalVolumeDisplay.ScaleTransferFunction = 'PiecewiseFunction'
invertedLocalVolumeDisplay.OpacityArray = ['POINTS', 'Q']
invertedLocalVolumeDisplay.OpacityTransferFunction = 'PiecewiseFunction'
invertedLocalVolumeDisplay.DataAxesGrid = 'GridAxesRepresentation'
invertedLocalVolumeDisplay.PolarAxes = 'PolarAxesRepresentation'
invertedLocalVolumeDisplay.ScalarOpacityUnitDistance = 0.12608545596058837
invertedLocalVolumeDisplay.OpacityArrayName = ['POINTS', 'Q']
invertedLocalVolumeDisplay.SelectInputVectors = ['POINTS', 'Normals']
invertedLocalVolumeDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
invertedLocalVolumeDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
invertedLocalVolumeDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from localVolumeBoundingBox
localVolumeBoundingBoxDisplay = Show(localVolumeBoundingBox, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
localVolumeBoundingBoxDisplay.Representation = 'Surface'
localVolumeBoundingBoxDisplay.AmbientColor = [0.0, 0.0, 1.0]
localVolumeBoundingBoxDisplay.ColorArrayName = [None, '']
localVolumeBoundingBoxDisplay.DiffuseColor = [0.0, 0.0, 1.0]
localVolumeBoundingBoxDisplay.LineWidth = 9.0
localVolumeBoundingBoxDisplay.RenderLinesAsTubes = 1
localVolumeBoundingBoxDisplay.SelectTCoordArray = 'None'
localVolumeBoundingBoxDisplay.SelectNormalArray = 'None'
localVolumeBoundingBoxDisplay.SelectTangentArray = 'None'
localVolumeBoundingBoxDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
localVolumeBoundingBoxDisplay.SelectOrientationVectors = 'None'
localVolumeBoundingBoxDisplay.ScaleFactor = 0.19982177942292767
localVolumeBoundingBoxDisplay.SelectScaleArray = 'None'
localVolumeBoundingBoxDisplay.GlyphType = 'Arrow'
localVolumeBoundingBoxDisplay.GlyphTableIndexArray = 'None'
localVolumeBoundingBoxDisplay.GaussianRadius = 0.009991088971146382
localVolumeBoundingBoxDisplay.SetScaleArray = [None, '']
localVolumeBoundingBoxDisplay.ScaleTransferFunction = 'PiecewiseFunction'
localVolumeBoundingBoxDisplay.OpacityArray = [None, '']
localVolumeBoundingBoxDisplay.OpacityTransferFunction = 'PiecewiseFunction'
localVolumeBoundingBoxDisplay.DataAxesGrid = 'GridAxesRepresentation'
localVolumeBoundingBoxDisplay.PolarAxes = 'PolarAxesRepresentation'
localVolumeBoundingBoxDisplay.SelectInputVectors = [None, '']
localVolumeBoundingBoxDisplay.WriteLog = ''

# show data from witnessBoundingBoxOutline
witnessBoundingBoxOutlineDisplay = Show(witnessBoundingBoxOutline, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
witnessBoundingBoxOutlineDisplay.Representation = 'Surface'
witnessBoundingBoxOutlineDisplay.AmbientColor = [1.0, 0.0, 0.0]
witnessBoundingBoxOutlineDisplay.ColorArrayName = [None, '']
witnessBoundingBoxOutlineDisplay.DiffuseColor = [1.0, 0.0, 0.0]
witnessBoundingBoxOutlineDisplay.LineWidth = 6.0
witnessBoundingBoxOutlineDisplay.RenderLinesAsTubes = 1
witnessBoundingBoxOutlineDisplay.SelectTCoordArray = 'None'
witnessBoundingBoxOutlineDisplay.SelectNormalArray = 'None'
witnessBoundingBoxOutlineDisplay.SelectTangentArray = 'None'
witnessBoundingBoxOutlineDisplay.BackfaceAmbientColor = [0.788235294117647, 0.788235294117647, 0.788235294117647]
witnessBoundingBoxOutlineDisplay.BackfaceDiffuseColor = [0.6196078431372549, 0.6196078431372549, 0.6196078431372549]
witnessBoundingBoxOutlineDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
witnessBoundingBoxOutlineDisplay.SelectOrientationVectors = 'None'
witnessBoundingBoxOutlineDisplay.ScaleFactor = 0.19982177942292767
witnessBoundingBoxOutlineDisplay.SelectScaleArray = 'None'
witnessBoundingBoxOutlineDisplay.GlyphType = 'Arrow'
witnessBoundingBoxOutlineDisplay.GlyphTableIndexArray = 'None'
witnessBoundingBoxOutlineDisplay.GaussianRadius = 0.009991088971146382
witnessBoundingBoxOutlineDisplay.SetScaleArray = [None, '']
witnessBoundingBoxOutlineDisplay.ScaleTransferFunction = 'PiecewiseFunction'
witnessBoundingBoxOutlineDisplay.OpacityArray = [None, '']
witnessBoundingBoxOutlineDisplay.OpacityTransferFunction = 'PiecewiseFunction'
witnessBoundingBoxOutlineDisplay.DataAxesGrid = 'GridAxesRepresentation'
witnessBoundingBoxOutlineDisplay.PolarAxes = 'PolarAxesRepresentation'
witnessBoundingBoxOutlineDisplay.SelectInputVectors = [None, '']
witnessBoundingBoxOutlineDisplay.WriteLog = ''

# show data from tableToPoints1
tableToPoints1Display = Show(tableToPoints1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
tableToPoints1Display.Representation = 'Points'
tableToPoints1Display.AmbientColor = [1.0, 0.0, 0.0]
tableToPoints1Display.ColorArrayName = [None, '']
tableToPoints1Display.DiffuseColor = [1.0, 0.0, 0.0]
tableToPoints1Display.PointSize = 25.0
tableToPoints1Display.RenderPointsAsSpheres = 1
tableToPoints1Display.SelectTCoordArray = 'None'
tableToPoints1Display.SelectNormalArray = 'None'
tableToPoints1Display.SelectTangentArray = 'None'
tableToPoints1Display.OSPRayScaleFunction = 'PiecewiseFunction'
tableToPoints1Display.SelectOrientationVectors = 'None'
tableToPoints1Display.ScaleFactor = 0.1575
tableToPoints1Display.SelectScaleArray = 'None'
tableToPoints1Display.GlyphType = 'Arrow'
tableToPoints1Display.GlyphTableIndexArray = 'None'
tableToPoints1Display.GaussianRadius = 0.007875
tableToPoints1Display.SetScaleArray = [None, '']
tableToPoints1Display.ScaleTransferFunction = 'PiecewiseFunction'
tableToPoints1Display.OpacityArray = [None, '']
tableToPoints1Display.OpacityTransferFunction = 'PiecewiseFunction'
tableToPoints1Display.DataAxesGrid = 'GridAxesRepresentation'
tableToPoints1Display.PolarAxes = 'PolarAxesRepresentation'
tableToPoints1Display.SelectInputVectors = [None, '']
tableToPoints1Display.WriteLog = ''

# show data from sideplanezyx0
sideplanezyx0Display = Show(sideplanezyx0, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sideplanezyx0Display.Representation = 'Surface'
sideplanezyx0Display.AmbientColor = [0.48627450980392156, 0.48627450980392156, 0.48627450980392156]
sideplanezyx0Display.ColorArrayName = ['POINTS', '']
sideplanezyx0Display.DiffuseColor = [0.48627450980392156, 0.48627450980392156, 0.48627450980392156]
sideplanezyx0Display.Opacity = 0.22
sideplanezyx0Display.SelectTCoordArray = 'None'
sideplanezyx0Display.SelectNormalArray = 'None'
sideplanezyx0Display.SelectTangentArray = 'None'
sideplanezyx0Display.OSPRayScaleArray = 'Q'
sideplanezyx0Display.OSPRayScaleFunction = 'PiecewiseFunction'
sideplanezyx0Display.SelectOrientationVectors = 'None'
sideplanezyx0Display.ScaleFactor = 0.2
sideplanezyx0Display.SelectScaleArray = 'Q'
sideplanezyx0Display.GlyphType = 'Arrow'
sideplanezyx0Display.GlyphTableIndexArray = 'Q'
sideplanezyx0Display.GaussianRadius = 0.01
sideplanezyx0Display.SetScaleArray = ['POINTS', 'Q']
sideplanezyx0Display.ScaleTransferFunction = 'PiecewiseFunction'
sideplanezyx0Display.OpacityArray = ['POINTS', 'Q']
sideplanezyx0Display.OpacityTransferFunction = 'PiecewiseFunction'
sideplanezyx0Display.DataAxesGrid = 'GridAxesRepresentation'
sideplanezyx0Display.PolarAxes = 'PolarAxesRepresentation'
sideplanezyx0Display.SelectInputVectors = [None, '']
sideplanezyx0Display.WriteLog = ''

# show data from bottomplanexzy0
bottomplanexzy0Display = Show(bottomplanexzy0, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
bottomplanexzy0Display.Representation = 'Surface'
bottomplanexzy0Display.AmbientColor = [0.41568627450980394, 0.41568627450980394, 0.41568627450980394]
bottomplanexzy0Display.ColorArrayName = ['POINTS', '']
bottomplanexzy0Display.DiffuseColor = [0.41568627450980394, 0.41568627450980394, 0.41568627450980394]
bottomplanexzy0Display.Opacity = 0.24
bottomplanexzy0Display.SelectTCoordArray = 'None'
bottomplanexzy0Display.SelectNormalArray = 'None'
bottomplanexzy0Display.SelectTangentArray = 'None'
bottomplanexzy0Display.OSPRayScaleArray = 'Q'
bottomplanexzy0Display.OSPRayScaleFunction = 'PiecewiseFunction'
bottomplanexzy0Display.SelectOrientationVectors = 'None'
bottomplanexzy0Display.ScaleFactor = 0.2670000076293945
bottomplanexzy0Display.SelectScaleArray = 'Q'
bottomplanexzy0Display.GlyphType = 'Arrow'
bottomplanexzy0Display.GlyphTableIndexArray = 'Q'
bottomplanexzy0Display.GaussianRadius = 0.013350000381469726
bottomplanexzy0Display.SetScaleArray = ['POINTS', 'Q']
bottomplanexzy0Display.ScaleTransferFunction = 'PiecewiseFunction'
bottomplanexzy0Display.OpacityArray = ['POINTS', 'Q']
bottomplanexzy0Display.OpacityTransferFunction = 'PiecewiseFunction'
bottomplanexzy0Display.DataAxesGrid = 'GridAxesRepresentation'
bottomplanexzy0Display.PolarAxes = 'PolarAxesRepresentation'
bottomplanexzy0Display.SelectInputVectors = [None, '']
bottomplanexzy0Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
bottomplanexzy0Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
bottomplanexzy0Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from backplanexyz0_1
backplanexyz0_1Display = Show(backplanexyz0_1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
backplanexyz0_1Display.Representation = 'Surface'
backplanexyz0_1Display.ColorArrayName = [None, '']
backplanexyz0_1Display.Opacity = 0.49
backplanexyz0_1Display.SelectTCoordArray = 'TextureCoordinates'
backplanexyz0_1Display.SelectNormalArray = 'Normals'
backplanexyz0_1Display.SelectTangentArray = 'None'
backplanexyz0_1Display.OSPRayScaleArray = 'Normals'
backplanexyz0_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
backplanexyz0_1Display.SelectOrientationVectors = 'None'
backplanexyz0_1Display.ScaleFactor = 0.534000015258789
backplanexyz0_1Display.SelectScaleArray = 'None'
backplanexyz0_1Display.GlyphType = 'Arrow'
backplanexyz0_1Display.GlyphTableIndexArray = 'None'
backplanexyz0_1Display.GaussianRadius = 0.026700000762939453
backplanexyz0_1Display.SetScaleArray = ['POINTS', 'Normals']
backplanexyz0_1Display.ScaleTransferFunction = 'PiecewiseFunction'
backplanexyz0_1Display.OpacityArray = ['POINTS', 'Normals']
backplanexyz0_1Display.OpacityTransferFunction = 'PiecewiseFunction'
backplanexyz0_1Display.DataAxesGrid = 'GridAxesRepresentation'
backplanexyz0_1Display.PolarAxes = 'PolarAxesRepresentation'
backplanexyz0_1Display.SelectInputVectors = ['POINTS', 'Normals']
backplanexyz0_1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
backplanexyz0_1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
backplanexyz0_1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# restore active source
SetActiveSource(None)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')