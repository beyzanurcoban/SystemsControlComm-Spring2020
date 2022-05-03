%
% File: DoublePendulum.m
%
classdef Antenna < handle
  % A class for implementing a MATLAB Graphics visualization of the
  % Simulink model of the double pendulum which Guy Rouleau shared in
  % File Exchange posting 23126.
  %
  properties (SetAccess=private)
    my_graphics    = gobjects(1,3);
    my_text  = gobjects(1,2);
    h_axes = hgtransform();
    angle = 0;
  end
  %
  % Public methods
  methods
    function obj = Antenna()
      obj.create_antenna();
    end
    
    function setAngle(obj, theta)
    % Call this to change the angles.
      obj.angle = theta;
      obj.transform();
    end

    function r = isAlive(obj)
    % Call this to check whether the figure window is still alive.
      r = isvalid(obj) && ...
          isvalid(obj.my_graphics(1)) && isvalid(obj.my_graphics(2)) &&  isvalid(obj.my_graphics(3)) &&...
          isvalid(obj.my_text(1)) && isvalid(obj.my_text(2));
    end
  end
  %
  % Private methods
  methods (Access=private)

    function transform(obj)
    % Updates the transform matrices.
    Rz = makehgtform('zrotate',obj.angle);
    % Scaling matrix
    % Concatenate the transforms and
    % set the transform Matrix property
    set(obj.h_axes,'Matrix',Rz)
    obj.my_text(2).String =[' Angle: ', num2str((obj.angle/pi)*180)];
    drawnow
    end
    
    function create_antenna(obj)
        close all;
        ax = axes('XLim',[-2 2],'YLim',[-2 2],'ZLim',[-1 1.5]);
        view(3)
        obj.h_axes = hgtransform('Parent',ax);
        hold on;

        [X1,Y1] = meshgrid(0:0.01:1.5, -1.5:0.01:1.5);
        Z1 = sqrt((1/7 + X1.^2/1+Y1.^2/1));
        surf1 = surf(X1,  Y1,  Z1,'Parent',  obj.h_axes);
        obj.my_graphics(1) = surf1;
        hold on;
        [X2,Y2] = meshgrid(-1.5:0.01:0, -1.5:0.01:1.5);
        Z2 = sqrt((1/7 + X2.^2/1+Y2.^2/1));
        surf2 = mesh(X2,  Y2 , Z2,'Parent', obj.h_axes , 'FaceColor','interp','FaceLighting','gouraud');
        obj.my_graphics(2) = surf2;
        hold on;
        
        [X3,Y3,Z3] = cylinder(0.4);
        Z3 = Z3 *0.5;
        disk1 = mesh(X3,  Y3 , Z3,'Parent',       obj.h_axes, 'FaceColor', [1,1,0]);

        obj.my_graphics(3) = disk1;
        
        obj.my_text(1) = text(1,1, -1.5,'Elec 304 Antenna ','HorizontalAlignment','center','FontSize',8);
        obj.my_text(2) = text(1, 1, - 1 , [' Angle: ', num2str((obj.angle/pi)*180)],'HorizontalAlignment','center','FontSize',12);
        
    end
  end
end