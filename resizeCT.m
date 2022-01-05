% This code is to resize CT .mat files

clc; clear all; close all;
name = 'vertebra20';
load(fullfile('orgCT', name))
org = eval(name);               % Convert str to variable

scaled = imresize3(org, [128, 128, 50]);

save(['resizedCT/', name, '.mat'], 'scaled')

%-------> Uncomment if you want to visulized the reshaped CT
% volumeViewer(scaled)

