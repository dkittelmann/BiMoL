%%%% Convert spmeeg file format into fieldtrip file format %%%%

ft_hastoolbox('spm12',1);

% Create id cell array
id = cell(1, 31);   
for i = 1:31
    id{i} = sprintf('%02d', i );   
end
id = string(id);

% Loop through all participants ids

for i=id
    if i == "20"
        continue;
    end    
    
    % change to participant dir
    participantDir = ['/Users/denisekittelmann/Documents/MATLAB/Hannah_data/EEG/P',char(i), '/'];
    cd(participantDir);

    % Find file and import file
    t = dir('eTadffspm*.mat');
    D = spm_eeg_load(t.name);

    % Convert file to fieldtrip
    fD = spm2fieldtrip(D);
    %fD.trialinfo = new_conds;
    %disp(fieldnames(fD))


    new_conds = cell(length(fD.trialinfo), 1);

    % Map fD.trialinfo to its corresponding condition label from D.condlist
    for j = 1:length(fD.trialinfo)
        cond_index = fD.trialinfo(j);  
        if cond_index <= length(D.condlist) && cond_index > 0
            new_conds{j} = D.condlist{cond_index};  % Map to condition name
        else
            warning(['Invalid condition index ', num2str(cond_index), ' in trialinfo for participant ', char(i)]);
            new_conds{j} = 'Unknown';  
        end
    end
    

    fD.trialinfo = new_conds;

    outputDir = fullfile(participantDir, 'fd');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);  

    end 
    
    % Save file 
    ft_file = fullfile(outputDir, ['eTadff_sub', char(i), '.mat']);
    save(ft_file, 'fD')

end


