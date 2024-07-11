//=============================================//
Mesh.MshFileVersion = 2.2; // KEEP THIS LINE 

// PARAMETERS
lc    = 0.05;
lcend = lc*10;
h     = 1.0;
Lx    = 40.0;
Ly    = 25.0;
Lz    = Pi;
D     = 1;
R     = D/2;

nb_inv = 10;

//cylinder params
Cx    = Ly*0.25;
Cy    = Ly*0.5;
lcyl  = 0.025;

// elements length
Npcyl  = 150;
Nex    = 70;
Npx    = Nex + 1;
Ne_in  = 40;
Np_in  = Ne_in + 1;
Ne_out = 70;
Np_out = Ne_out + 1;

delta_R      = R;

// refinement box WAKE params
VIn_scale = 2;
Vout_scale = 15;
xmin_refin = Cx;
xmax_refin = Cx + 20;
ymin_refin = Cy - 5*delta_R;
ymax_refin = Cy + 5*delta_R;
thick_refin = 5;

// refinement cylinder NEAR CYLINDER params
VIn_scale_2 = VIn_scale*0.8;
Vout_scale_2 = Vout_scale*0.9;

// refinement cylinder far CYLINDER params (not used)
VIn_scale_3 = VIn_scale*0.7;
Vout_scale_3 = Vout_scale*0.9;

// refinement box WAKE params
VIn_scale_4 = VIn_scale*0.5;
Vout_scale_4 = Vout_scale*0.5;
xmin_refin_4 = Cx;
xmax_refin_4 = Cx + 5;
ymin_refin_4 = Cy - 2*delta_R;
ymax_refin_4 = Cy + 2*delta_R;
thick_refin_4 = thick_refin*2;


// params jets
alpha = 10*Pi/180;
Npjets = 20;

// extrude params

Nez_lay = 12;

Mesh.ElementOrder = 1;

// MAIN FACE
//domain
Point(1) = {0 , 0  , 0 , lc};
Point(2) = {Lx, 0  , 0 , lcend};
Point(3) = {Lx, Ly , 0 , lcend};
Point(4) = {0 , Ly , 0 , lc};

//cylinder
Point(5) = {Cx   , Cy   , 0 , lcyl}; //center
Point(6) = {Cx   , Cy+R , 0 , lcyl}; //12h
Point(7) = {Cx+R , Cy   , 0 , lcyl}; //3h
Point(8) = {Cx   , Cy-R , 0 , lcyl}; //6h
Point(9) = {Cx-R , Cy   , 0 , lcyl}; //9h

//JETS (will change depending on jets.py locations ??)
Point(100) = {Cx - R*Sin(alpha/2),  Cy + R*Cos(alpha/2), 0 , lcyl}; //top left 
Point(101) = {Cx + R*Sin(alpha/2),  Cy + R*Cos(alpha/2), 0 , lcyl}; //top right
Point(200) = {Cx - R*Sin(alpha/2),  Cy - R*Cos(alpha/2), 0 , lcyl}; //bot left
Point(201) = {Cx + R*Sin(alpha/2),  Cy - R*Cos(alpha/2), 0 , lcyl}; //bot right

// block loop
Line(1001)     = {1, 2};
Line(1002)     = {2, 3};
Line(1003)     = {3, 4};
Line(1004)     = {4, 1};

//Cylinder surface no-slip
Circle(1005)   = {101, 5, 7};
Circle(1006)   = {7, 5, 201};
Circle(1007)   = {200, 5, 9};
Circle(1008)   = {9, 5, 100};

//Cylinder jets
Circle(10001)  = {100, 5, 101};
Circle(10002)  = {201, 5, 200};

// elements per surface
Transfinite Line{1001} = Npx;
Transfinite Line{1002} = Np_out;
Transfinite Line{1003} = Npx;
Transfinite Line{1004} = Np_in;
Transfinite Line{1005} = Npcyl;
Transfinite Line{1006} = Npcyl;
Transfinite Line{1007} = Npcyl;
Transfinite Line{1008} = Npcyl;

Transfinite Line{10001} = Npjets;
Transfinite Line{10002} = Npjets;



Line        Loop(601)     = {1001, 1002, 1003, 1004};
Line        Loop(602)     = {1005, 1006, 10002, 1007, 1008, 10001};

Plane       Surface(2001) = {601, 602};
//Transfinite Surface{2001} = {1, 2, 3, 4};
//Recombine   Surface{2001}; //QUA

////=============================================//
////=============================================//


vol1[] = Extrude {0, 0, Lz/nb_inv} { Surface{2001}; Layers{Nez_lay}; Recombine; };
vol2[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol1[0]}; Layers{Nez_lay}; Recombine; };
vol3[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol2[0]}; Layers{Nez_lay}; Recombine; };
vol4[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol3[0]}; Layers{Nez_lay}; Recombine; };
vol5[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol4[0]}; Layers{Nez_lay}; Recombine; };
vol6[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol5[0]}; Layers{Nez_lay}; Recombine; };
vol7[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol6[0]}; Layers{Nez_lay}; Recombine; };
vol8[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol7[0]}; Layers{Nez_lay}; Recombine; };
vol9[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol8[0]}; Layers{Nez_lay}; Recombine; };
vol10[] =  Extrude {0, 0, Lz/nb_inv} { Surface{vol9[0]}; Layers{Nez_lay}; Recombine; };

Field[1] = Box;
Field[1].VIn = lc*VIn_scale; //2.5;
Field[1].VOut = lc*Vout_scale; //7;
Field[1].XMin = xmin_refin; //6.5;
Field[1].XMax = xmax_refin; //25;
Field[1].YMin = ymin_refin; //5;
Field[1].YMax = ymax_refin; //10;
Field[1].ZMin = 0;
Field[1].ZMax = Lz;
Field[1].Thickness = thick_refin; //7;

Field[2] = Box;
Field[2].VIn = lc*VIn_scale_4; //2.5;
Field[2].VOut = lc*Vout_scale_4; //7;
Field[2].XMin = xmin_refin_4; //6.5;
Field[2].XMax = xmax_refin_4; //25;
Field[2].YMin = ymin_refin_4; //5;
Field[2].YMax = ymax_refin_4; //10;
Field[2].ZMin = 0;
Field[2].ZMax = Lz;
Field[2].Thickness = thick_refin_4; //7;


Field[3] = Cylinder;
Field[3].VIn = lc*VIn_scale_2;
Field[3].VOut = lc*Vout_scale_2; 
Field[3].Radius = R+delta_R;
Field[3].XCenter = Cx;
Field[3].YCenter = Cy;
Field[3].ZCenter = 0;
Field[3].ZAxis = Lz;

// NOT USED ////////////////////////////7
Field[4] = Cylinder;
Field[4].VIn = lc*VIn_scale_3; //2.5;
Field[4].VOut = lc*Vout_scale_3; //7;
Field[4].Radius = R+3*delta_R; //6.5;
Field[4].XCenter = Cx;
Field[4].YCenter = Cy;
Field[4].ZCenter = 0;
Field[4].ZAxis = Lz;
//////////////////////////////////////

Field[5] = Min;
Field[5].FieldsList = {1, 2, 3};
Background Field = 5;

Transfinite       Volume{vol1[1],vol2[1],vol3[1],vol4[1],vol5[1],vol6[1],vol7[1],vol8[1],vol9[1],vol10[1]};
Recombine         Volume{vol1[1],vol2[1],vol3[1],vol4[1],vol5[1],vol6[1],vol7[1],vol8[1],vol9[1],vol10[1]};

//Mesh.Algorithm3D = 1;

//=============================================//
//=============================================//

//// PHYSICAL VARIABLES
Physical Surface ("front",    1) = {2001};
Physical Surface ("back",     2) = {vol10[0]};
Physical Surface ("top",      3) = {vol1[2],vol2[2],vol3[2],vol4[2],vol5[2],vol6[2],vol7[2],vol8[2],vol9[2],vol10[2]};
Physical Surface ("out",      4) = {vol1[3],vol2[3],vol3[3],vol4[3],vol5[3],vol6[3],vol7[3],vol8[3],vol9[3], vol10[3]};
Physical Surface ("bot",      5) = {vol1[4],vol2[4],vol3[4],vol4[4],vol5[4],vol6[4],vol7[4],vol8[4],vol9[4], vol10[4]};
Physical Surface ("in",       6) = {vol1[5],vol2[5],vol3[5],vol4[5],vol5[5],vol6[5],vol7[5],vol8[5],vol9[5], vol10[5]};
Physical Surface ("cylinder_inv1", 7) = {vol1[7],vol1[8],vol1[10],vol1[11]};
Physical Surface ("cylinder_inv2", 8) = {vol2[7],vol2[8],vol2[10],vol2[11]};
Physical Surface ("cylinder_inv3", 9) = {vol3[7],vol3[8],vol3[10],vol3[11]};
Physical Surface ("cylinder_inv4", 10) = {vol4[7],vol4[8],vol4[10],vol4[11]};
Physical Surface ("cylinder_inv5", 11) = {vol5[7],vol5[8],vol5[10],vol5[11]};
Physical Surface ("cylinder_inv6", 12) = {vol6[7],vol6[8],vol6[10],vol6[11]};
Physical Surface ("cylinder_inv7", 13) = {vol7[7],vol7[8],vol7[10],vol7[11]};
Physical Surface ("cylinder_inv8", 14) = {vol8[7],vol8[8],vol8[10],vol8[11]};
Physical Surface ("cylinder_inv9", 15) = {vol9[7],vol9[8],vol9[10],vol9[11]};
Physical Surface ("cylinder_inv10", 16) = {vol10[7],vol10[8],vol10[10],vol10[11]};

Physical Surface ("jet_top",  17) = {vol1[6],vol2[6],vol3[6],vol4[6],vol5[6],vol6[6],vol7[6],vol8[6],vol9[6], vol10[6]};
Physical Surface ("jet_bot",  18) = {vol1[9],vol2[9],vol3[9],vol4[9],vol5[9],vol6[9],vol7[9],vol8[9],vol9[9], vol10[9]};
Physical Volume  ("fluid",    19) = {vol1[1],vol2[1],vol3[1],vol4[1],vol5[1],vol6[1],vol7[1],vol8[1],vol9[1], vol10[1]};


//=============================================//
//=============================================//

// GUI SETTINGS
Geometry.PointNumbers   = 1;
Geometry.LineNumbers    = 1;
Geometry.SurfaceNumbers = 1;
Geometry.Color.Points   = {0,150,0};
General.Color.Text      = White;
Mesh.Color.Points       = {255,0,255};
