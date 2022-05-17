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
 <img src="./images/fft.png" width = "100%" height = "100%" alt="FFT" align=center />
```Matlab
 % *%TODO* :
%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
Mix = reshape(Mix, [Nr, Nd]);
 % *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
signal_fft = fft(Mix, Nr);
 % *%TODO* :
% Take the absolute value of FFT output
signal_fft = abs(signal_fft);
signal_fft = signal_fft ./ max(signal_fft);
 % *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
signal_fft = signal_fft(1 : Nr/2-1);
```

#### 4. 2D CFAR
Implement the 2D CFAR process on the output of 2D FFT operation, i.e the Range Doppler Map.
```Matlab
% *%TODO* :
%Select the number of Training Cells in both the dimensions.
n_train_cells = 10;
n_train_bands = 8;
% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
n_guard_cells = 4;
n_guard_bands = 4;
% *%TODO* :
% offset the threshold by SNR value in dB
offset = 1.37;
% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(1,1);


% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.


   % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
   % CFAR
RDM = RDM / max(RDM(:));
for row0 = n_train_cells + n_guard_cells + 1 : (Nr/2) - (n_train_cells + n_guard_cells)
  for col0 = n_train_bands + n_guard_bands + 1 : (Nd) - (n_train_bands + n_guard_bands)
    %Create a vector to store noise_level for each iteration on training cells
    noise_level = zeros(1, 1);

    for row1 = row0 - (n_train_cells + n_guard_cells) : row0 + (n_train_cells + n_guard_cells)
      for col1 = col0 - (n_train_bands + n_guard_bands) : col0 + (n_train_bands + n_guard_bands)
        if (abs(row0 - row1) > n_guard_cells || abs(col0 - col1) > n_guard_bands)
          noise_level = noise_level + db2pow(RDM(row1, col1));
        end
      end
    end

    % Calculate threshold from noise average then add the offset
    thresh = pow2db(noise_level / (2 * (n_train_bands + n_guard_bands + 1) * 2 * (n_train_cells + n_guard_cells + 1) - (n_guard_cells * n_guard_bands) - 1));
    thresh = thresh + offset;

    CUT = RDM(row1,col1);

    if (CUT < thresh)
      RDM(row0, col0) = 0;
    else
      RDM(row0, col0) = 1;
    end

  end
end
'''
