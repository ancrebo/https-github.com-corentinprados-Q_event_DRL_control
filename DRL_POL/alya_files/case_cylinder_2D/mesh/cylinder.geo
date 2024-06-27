// Define Domain Points
Point(1) = {0, 0, 0, 0.19130434782608696};
Point(2) = {22.0, 0, 0, 1.1478260869565218};
Point(3) = {22.0, 4.1, 0, 1.1478260869565218};
Point(4) = {0, 4.1, 0, 0.19130434782608696};


// Define Cylinder Points
Point(5) = {2.0, 2.0, 0, 0.0076521739130434785};
Point(6) = {2.0, 2.5, 0, 0.0076521739130434785};
Point(7) = {1.5, 2.0, 0, 0.0076521739130434785};
Point(8) = {2.0, 1.5, 0, 0.0076521739130434785};
Point(9) = {2.5, 2.0, 0, 0.0076521739130434785};
Point(10) = {1.956422128626171, 2.4980973490458727, 0, 0.0076521739130434785};
Point(11) = {2.043577871373829, 2.4980973490458727, 0, 0.0076521739130434785};
Point(12) = {2.043577871373829, 1.5019026509541273, 0, 0.0076521739130434785};
Point(13) = {1.956422128626171, 1.5019026509541273, 0, 0.0076521739130434785};

// Define Karman Points
Point(14) = {12, 4.1, 0, 0.31428571428571433};
Point(15) = {12, 0, 0, 0.31428571428571433};


// Draw the cylinder
Circle(1) = { 12, 5, 8};
Circle(2) = { 8, 5, 13};
Circle(3) = { 13, 5, 7};
Circle(4) = { 7, 5, 10};
Circle(5) = { 10, 5, 6};
Circle(6) = { 6, 5, 11};
Circle(7) = { 11, 5, 9};
Circle(8) = { 9, 5, 12};


// Define the block domain line
Line(10) = {4, 14};
Line(11) = {14, 3};
Line(12) = {3, 2};
Line(13) = {2, 15};
Line(14) = {15, 1};
Line(15) = {1, 4};


// Define the Line Loop
Line Loop(16) = {1, 2, 3, 4, 5, 6, 7, 8};
Line Loop(17) = {10, 11, 12, 13, 14, 15};


// Define the Surfaces
Plane Surface(2) = {16, 17};


// Create Physical surface
Physical Curve ("inlet") = {15};
Physical Curve ("outlet") = {12};
Physical Curve ("cylinder_wall") = {3,4,7,8};
Physical Curve ("top") = {10,11};
Physical Curve ("bottom") = {13,14};
Physical Curve ("jet_bottom") = {1,2};
Physical Curve ("jet_top") = {5,6};
Physical Surface ("fluid") = {2,3};
