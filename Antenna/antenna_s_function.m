function antenna_s_function(block)
% Level-2 MATLAB file S-Function for continuous time variable step demo.
%   Copyright 1990-2009 The MathWorks, Inc.

  setup(block);
  
%endfunction

function setup(block)
  
  %% Register number of input and output ports
  block.NumInputPorts  = 1;
  block.NumOutputPorts = 0;

  %% Setup functional port properties to dynamically
  %% inherited.
  block.SetPreCompInpPortInfoToDynamic;

  %% Set block sample time to variable sample time
  block.NumDialogPrms     = 0;

% Register sample times
%  [0 offset]            : Continuous sample time
%  [positive_num offset] : Discrete sample time
%
%  [-1, 0]               : Inherited sample time
%  [-2, 0]               : Variable sample time
  block.SampleTimes = [-1 0];
  block.SetSimViewingDevice(true)
  %% Set the block simStateCompliance to default (i.e., same as a built-in block)
block.SimStateCompliance = 'DefaultSimState';

 %% Register methods

block.RegBlockMethod('PostPropagationSetup',    @DoPostPropSetup);
block.RegBlockMethod('InitializeConditions', @InitializeConditions);
block.RegBlockMethod('Start', @Start);
block.RegBlockMethod('Update', @Update);
block.RegBlockMethod('Derivatives', @Derivatives);
block.RegBlockMethod('Terminate', @Terminate); % Required

%endfunction
%%
%% PostPropagationSetup:
%%   Functionality    : Setup work areas and state variables. Can
%%                      also register run-time methods here
%%   Required         : No
%%   C-Mex counterpart: mdlSetWorkWidths
%%
function DoPostPropSetup(block)
  block.NumDworks = 1;
  
  block.Dwork(1).Name            = 'x1';
  block.Dwork(1).Dimensions      = 1;
  block.Dwork(1).DatatypeID      = 0;      % double
  block.Dwork(1).Complexity      = 'Real'; % real
  block.Dwork(1).UsedAsDiscState = true;
 

%%
%% InitializeConditions:
%%   Functionality    : Called at the start of simulation and if it is 
%%                      present in an enabled subsystem configured to reset 
%%                      states, it will be called when the enabled subsystem
%%                      restarts execution to reset the states.
%%   Required         : No
%%   C-MEX counterpart: mdlInitializeConditions
%%
function InitializeConditions(block)
ud = Antenna();
  
  %
  % Save it in UserData
 set_param(block.BlockHandle,'UserData',ud);
% ax = axes('XLim',[-1.5 1.5],'YLim',[-1.5 1.5],'ZLim',[0 1.5]);
% view(3)
% h = hgtransform('Parent',ax);
% hold on;
% 
% [X1,Y1] = meshgrid(0:0.01:1.5, -1.5:0.01:1.5);
% Z1 = sqrt((1/7 + X1.^2/1+Y1.^2/1));
% mesh(X1,  Y1,  Z1,'Parent',h, 'FaceColor', [1,0,0]);
% 
% hold on;
% [X2,Y2] = meshgrid(-1.5:0.01:0, -1.5:0.01:1.5);
% Z2 = sqrt((1/7 + X2.^2/1+Y2.^2/1));
% mesh(X2,  Y2 , Z2,'Parent',h, 'FaceColor', [1,1,0]);
% 
% hold on;
% [X3,Y3,Z3] = cylinder(0.4);
% Z3 = Z3 *0.5;
% mesh(X3,  Y3 , Z3,'Parent',h, 'FaceColor', [1,1,1]);
% 
% set_param(block.BlockHandle, 'UserData', h);
%end InitializeConditions


%%
%% Start:
%%   Functionality    : Called once at start of model execution. If you
%%                      have states that should be initialized once, this 
%%                      is the place to do it.
%%   Required         : No
%%   C-MEX counterpart: mdlStart
%%
function Start(block)

block.Dwork(1).Data = 0;
  %
  % Check to see if we already have an instance of DoublePendulum


%end Start

%%
%% Update:
%%   Functionality    : Called to update discrete states
%%                      during simulation step
%%   Required         : No
%%   C-MEX counterpart: mdlUpdate
%%
function Update(block)
block.Dwork(1).Data = block.InputPort(1).Data;
antenna = get_param(block.BlockHandle, 'UserData');
if(~isempty(antenna) && antenna.isAlive())
antenna.setAngle(block.Dwork(1).Data)
end
%end Update

%%
function Terminate(block)

%end Terminate


