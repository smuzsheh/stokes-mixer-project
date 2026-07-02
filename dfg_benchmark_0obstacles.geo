// Empty Channel - 0 Obstacles (Reference Case)
L = 2.2;
H = 0.41;

// Mesh sizes - SAME for all cases!
h_in = 0.01;
h_out = 0.02;

// Channel Walls
Point(1) = {0, 0, 0, h_in};
Point(2) = {L, 0, 0, h_out};
Point(3) = {L, H, 0, h_out};
Point(4) = {0, H, 0, h_in};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups
Physical Surface(1) = {1};
Physical Line(1) = {4};
Physical Line(2) = {1, 3};
Physical Line(3) = {};
Physical Line(4) = {2};

Mesh.Algorithm = 6;
Mesh.MshFileVersion = 4.1;