Mesh.MshFileVersion = 2.2;
//=============================================//
// PARAMETERS
lc    = 1.0e+2;

//h     = 1.0;
//Lx    = 4.0*Pi*h;
//Ly    = 2.0*h;
//Lz    = (4/3)*Pi*h;

h = 1; // # Common practise as Chebysev polynom but be aware that sometimes it will be 2 or something else (like in SIMSOn)
Lx = 2.67*h; // Length of the domain
Ly = 2*h; // Height of the domain 
Lz = 0.8*h; // Width of the domain 

Nex   = 48;
Npx   = Nex + 1;
Ney   = 192;
Npy   = Ney + 1;
Nez   = 48;
Npz   = Nez + 1;
alpha = 0.957; // for y+ = 1.0

Mesh.ElementOrder = 1;

atanhAlpha = 0;
For i In {1 : 3000}
  atanhAlpha = atanhAlpha + alpha^(2*i-1)/(2*i-1);
EndFor

Printf('%f',atanhAlpha);



// bottom
Point(1) = {0 , 0 , 0 , lc};
Point(2) = {Lx, 0 , 0 , lc};
Point(3) = {Lx, 0 , Lz, lc};
Point(4) = {0 , 0 , Lz, lc};


// bottom loop
Line(1001) = {1,2};
Line(1002) = {2,3};
Line(1003) = {3,4};
Line(1004) = {4,1};
Transfinite Line{1001} = Npx;
Transfinite Line{1002} = Npz;
Transfinite Line{1003} = Npx;
Transfinite Line{1004} = Npz;

// bottom face
Line        Loop(601)     = {1001,1002,1003,1004};
Plane       Surface(2001) = {601};
Transfinite Surface{2001} = {1,2,3,4};
Recombine   Surface{2001}; 


////=============================================//
////=============================================//
For i In {1 : Ney} 
    yupper = 0.5 *    (
               1.0+1.0/alpha*
               Tanh((-1.0+2.0*(i)/(Npy-1))*atanhAlpha)
               );					// normalised to restrict the values between 0 and 1
    NperLayer[i-1]  = 1;
    Printf('%f', yupper);
    y[i-1]          = yupper;
    
EndFor


out[]=Extrude {0, Ly, 0} { Surface{2001}; Layers{{NperLayer[]},{y[]}}; Recombine; };

Transfinite       Volume{out[1]};
Recombine         Volume{out[1]};  
Mesh.Algorithm3D = 1;
//=============================================//
//=============================================//
// PHYSICAL VARIABLES
Physical Surface ("bottom",1) = {2001};
Physical Surface ("top",2) = {out[0]};
Physical Surface ("back",3) = {out[2]};
Physical Surface ("right",4) = {out[3]};
Physical Surface ("front",5) = {out[4]};
Physical Surface ("left",6) = {out[5]};
Physical Volume  ("internalVolume",7) = {out[1]};


//=============================================//
//=============================================//
// GUI SETTINGS
Geometry.PointNumbers = 1;
Geometry.LineNumbers = 1;
Geometry.Color.Points = {0,150,0};
General.Color.Text = White;
Mesh.Color.Points = {255,0,255};
