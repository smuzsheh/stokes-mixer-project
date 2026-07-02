// Kenics Mixer - 6 Staggered Circles
L = 2.2;
H = 0.41;
r = 0.04;

// Mesh sizes - SAME for all cases!
h_in = 0.01;
h_cyl = 0.015;   // SAME for all cases!
h_out = 0.02;

// 6 element positions
cx1 = 0.30;  cy1 = 0.20;
cx2 = 0.55;  cy2 = 0.27;
cx3 = 0.80;  cy3 = 0.13;
cx4 = 1.05;  cy4 = 0.20;
cx5 = 1.30;  cy5 = 0.27;
cx6 = 1.55;  cy6 = 0.13;

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

// Circle 3
Point(30) = {cx3, cy3, 0, h_cyl};
Point(31) = {cx3 + r, cy3, 0, h_cyl};
Point(32) = {cx3, cy3 + r, 0, h_cyl};
Point(33) = {cx3 - r, cy3, 0, h_cyl};
Point(34) = {cx3, cy3 - r, 0, h_cyl};
Circle(30) = {31, 30, 32};
Circle(31) = {32, 30, 33};
Circle(32) = {33, 30, 34};
Circle(33) = {34, 30, 31};

// Circle 4
Point(40) = {cx4, cy4, 0, h_cyl};
Point(41) = {cx4 + r, cy4, 0, h_cyl};
Point(42) = {cx4, cy4 + r, 0, h_cyl};
Point(43) = {cx4 - r, cy4, 0, h_cyl};
Point(44) = {cx4, cy4 - r, 0, h_cyl};
Circle(40) = {41, 40, 42};
Circle(41) = {42, 40, 43};
Circle(42) = {43, 40, 44};
Circle(43) = {44, 40, 41};

// Circle 5
Point(50) = {cx5, cy5, 0, h_cyl};
Point(51) = {cx5 + r, cy5, 0, h_cyl};
Point(52) = {cx5, cy5 + r, 0, h_cyl};
Point(53) = {cx5 - r, cy5, 0, h_cyl};
Point(54) = {cx5, cy5 - r, 0, h_cyl};
Circle(50) = {51, 50, 52};
Circle(51) = {52, 50, 53};
Circle(52) = {53, 50, 54};
Circle(53) = {54, 50, 51};

// Circle 6
Point(60) = {cx6, cy6, 0, h_cyl};
Point(61) = {cx6 + r, cy6, 0, h_cyl};
Point(62) = {cx6, cy6 + r, 0, h_cyl};
Point(63) = {cx6 - r, cy6, 0, h_cyl};
Point(64) = {cx6, cy6 - r, 0, h_cyl};
Circle(60) = {61, 60, 62};
Circle(61) = {62, 60, 63};
Circle(62) = {63, 60, 64};
Circle(63) = {64, 60, 61};

// Surface
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {10,11,12,13};
Line Loop(3) = {20,21,22,23};
Line Loop(4) = {30,31,32,33};
Line Loop(5) = {40,41,42,43};
Line Loop(6) = {50,51,52,53};
Line Loop(7) = {60,61,62,63};
Plane Surface(1) = {1,2,3,4,5,6,7};

// Physical groups
Physical Surface(1) = {1};
Physical Line(1) = {4};
Physical Line(2) = {1, 3};
Physical Line(3) = {10,11,12,13,20,21,22,23,30,31,32,33,40,41,42,43,50,51,52,53,60,61,62,63};
Physical Line(4) = {2};

Mesh.Algorithm = 6;
Mesh.MshFileVersion = 4.1;