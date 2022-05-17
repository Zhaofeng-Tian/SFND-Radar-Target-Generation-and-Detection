# SFND-Radar-Target-Generation-and-Detection

---
#### 1. FMCW Waveform Design
Using the given system requirements, design a FMCW waveform. Find its Bandwidth (B), chirp time (Tchirp) and slope of the chirp.

```Matlab
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant
c = 3*10^8;
range = 110;
vel = -20;
max_range = 200;
range_res = 1;
max_vel = 100; 

%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.
B = c / (2*range_res);
Tchirp = 5.5 * 2 * (max_range/c);  
slope = B/Tchirp;
```

#### 2. Simulation Loop
Simulate Target movement and calculate the beat or mixed signal for every timestamp.

```Matlab
%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq

                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples


%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    
    
    % *%TODO* :
    %For each time stamp update the Range of the Target for constant velocity. 
    r_t(i) = range + (vel*t(i));
    td(i) = (2 * r_t(i)) / c;
    % *%TODO* :
    %For each time sample we need update the transmitted and
    %received signal. 
    Tx(i) = cos(2*pi*(fc*t(i) + (slope*t(i)^2)/2 ) );
    Rx (i)  = cos(2*pi*(fc*(t(i) -td(i)) + (slope * (t(i)-td(i))^2)/2 ) );
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i) .* Rx(i);
    
end

```

#### 3. Range FFT (1st FFT)

Implement the Range FFT on the Beat or Mixed Signal and plot the result.

```Matlab
signal = reshape(Mix,[Nr, Nd])
signal_fft = fft(signal, Nr)/Nr;
signal_fft = abs(signal_fft);

signal_fft  = signal_fft(1:Nr/2)   
disp(length(signal_fft))

figure ('Name','Range from First FFT')
subplot(2,1,1)
plot(signal_fft) 
axis ([0 200 0 0.5]);
```

#### 4. 2D CFAR
Implement the 2D CFAR process on the output of 2D FFT operation, i.e the Range Doppler Map.

Determine the number of Training cells for each dimension. Similarly, pick the number of guard cells.

```Matlab
Tr = 8;
Td = 4;
Gr = 8;
Gd = 4;
offset=1.35;
```

Slide the cell under test across the complete matrix. Make sure the CUT has margin for Training and Guard cells from the edges.

```Matlab
for i = 1:(Nr/2-(2*Gr+2*Tr+1))
    for j = 1:(Nd-(2*Gd+2*Td+1))
        ...
    end
end
```

For every iteration sum the signal level within all the training cells. To sum convert the value from logarithmic to linear using db2pow function.

```Matlab
noise_level = zeros(1,1);
for x = i:(i+2*Tr+2*Gr) 
    noise_level = [noise_level, db2pow(RDM(x,(j:(j+ 2*Td+2*Gd))))];
end    
sum_cell = sum(noise_level);
noise_level = zeros(1,1);
for x = (i+Tr):(i+Tr+2*Gr) 
    noise_level = [noise_level, db2pow(RDM(x,(j+Td):(j+Td+2*Gd)))];
end    
sum_guard = sum(noise_level);
sum_train = sum_cell - sum_guard;
```

Average thesummed values for all of the training cells used. After averaging convert it back to logarithmic using pow2db.
Further add the offset to it to determine the threshold.

```Matlab
threshold = pow2db(sum_train/Tcell)*offset;
```

Next, compare the signal under CUT against this threshold.
If the CUT level > threshold assign it a value of 1, else equate it to 0.

```Matlab
signal = RDM(i+Tr+Gr, j+Td+Gd);
if (signal < threshold)
    signal = 0;
else
    signal = 1;
end    
CFAR(i+Tr+Gr, j+Td+Gd) = signal;
```

To keep the map size same as it was before CFAR, equate all the non-thresholded cells to 0.

```Matlab
for i = 1:(Nr/2)
    for j = 1:Nd
       if (i > (Tr+Gr))& (i < (Nr/2-(Tr+Gr))) & (j > (Td+Gd)) & (j < (Nd-(Td+Gd)))
           continue
       end
       CFAR(i,j) = 0;
    end
end
```
Selection of Training, Guard cells and offset.

Training, Guard cells and offset are selected by increasing and decreasing to match the image shared in walkthrough.

Tr = 8;
Td = 4;
Gr = 8;
Gd = 4;
offset=1.35;

Result
