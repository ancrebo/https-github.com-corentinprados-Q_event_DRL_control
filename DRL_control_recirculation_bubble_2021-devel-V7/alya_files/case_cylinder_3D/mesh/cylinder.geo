//=============================================//
Mesh.MshFileVersion = 2.2; // KEEP THIS LINE 

// PARAMETERS
lc    = 0.05;
lcend = lc*10;
h     = 1.0;
Lx    = 30.0;
Ly    = 15;
Lz    = 2*Pi;
D     = 1;
R     = D/2;

//cylinder params
Cx    = Ly*0.5;
Cy    = Ly*0.5;
lcyl  = 0.025;

// elements length
Npcyl  = 75;
Nex    = 120;
Npx    = Nex + 1;
Ne_in  = 50;
Np_in  = Ne_in + 1;
Ne_out = 70;
Np_out = Ne_out + 1;


// params jets
alpha = 10*Pi/180;
Npjets = 20;

// extrude params
Nez_lay = 30;

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
Recombine   Surface{2001}; //QUA

////=============================================//
////=============================================//


vol[] = Extrude {0, 0, Lz} { Surface{2001}; Layers{Nez_lay}; Recombine; };

Transfinite       Volume{vol[1]};
Recombine         Volume{vol[1]};  
Mesh.Algorithm3D = 1;

//=============================================//
//=============================================//

// PHYSICAL VARIABLES
Physical Surface ("front",    1) = {2001};
Physical Surface ("back",     2) = {vol[0]};
Physical Surface ("top",      3) = {vol[2]};
Physical Surface ("out",      4) = {vol[3]};
Physical Surface ("bot",      5) = {vol[4]};
Physical Surface ("in",       6) = {vol[5]};
Physical Surface ("cylinder", 7) = {vol[7], vol[8], vol[10], vol[11]};
Physical Surface ("jet_top",  8) = {vol[6]};
Physical Surface ("jet_bot",  9) = {vol[9]};
Physical Volume  ("fluid",    10) = {vol[1]};


//=============================================//
//=============================================//

// GUI SETTINGS
Geometry.PointNumbers   = 1;
Geometry.LineNumbers    = 1;
Geometry.SurfaceNumbers = 1;
Geometry.Color.Points   = {0,150,0};
General.Color.Text      = White;
Mesh.Color.Points       = {255,0,255};
