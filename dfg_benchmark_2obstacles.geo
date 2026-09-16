// Kenics Mixer - 2 Staggered Circles
L = 2.2;
H = 0.41;
r = 0.04;

// Mesh sizes - SAME for all cases!
h_in = 0.01;
h_cyl = 0.015;
h_out = 0.02;

// 2 element positions
cx1 = 0.55;  cy1 = 0.20;
cx2 = 1.45;  cy2 = 0.20;

// Channel
Point(1) = {0, 0, 0, h_in};
Point(2) = {L, 0, 0, h_out};
Point(3) = {L, H, 0, h_out};
Point(4) = {0, H, 0, h_in};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Circle 1
Point(10) = {cx1, cy1, 0, h_cyl};
Point(11) = {cx1 + r, cy1, 0, h_cyl};
Point(12) = {cx1, cy1 + r, 0, h_cyl};
Point(13) = {cx1 - r, cy1, 0, h_cyl};
Point(14) = {cx1, cy1 - r, 0, h_cyl};
Circle(10) = {11, 10, 12};
Circle(11) = {12, 10, 13};
Circle(12) = {13, 10, 14};
Circle(13) = {14, 10, 11};

// Circle 2
Point(20) = {cx2, cy2, 0, h_cyl};
Point(21) = {cx2 + r, cy2, 0, h_cyl};
Point(22) = {cx2, cy2 + r, 0, h_cyl};
Point(23) = {cx2 - r, cy2, 0, h_cyl};
Point(24) = {cx2, cy2 - r, 0, h_cyl};
Circle(20) = {21, 20, 22};
Circle(21) = {22, 20, 23};
Circle(22) = {23, 20, 24};
Circle(23) = {24, 20, 21};

// Surface
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {10,11,12,13};
Line Loop(3) = {20,21,22,23};
Plane Surface(1) = {1,2,3};

// Physical groups
Physical Surface(1) = {1};
Physical Line(1) = {4};
Physical Line(2) = {1, 3};
Physical Line(3) = {10,11,12,13,20,21,22,23};
Physical Line(4) = {2};

Mesh.Algorithm = 6;
Mesh.MshFileVersion = 4.1;