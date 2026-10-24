function export_nitin_predictions(evaluationMat, manifestCsv, outputJson)
% Convert R.Predictions from Nitin's evaluation.mat to common COCO JSON.
% Run in MATLAB; check that boxes are [x,y,width,height] as expected.
% Example:
% export_nitin_predictions('evaluation.mat','split_manifest.csv','yolov4_val_predictions.json')

loaded = load(evaluationMat, 'R');
if ~isfield(loaded, 'R') || ~isfield(loaded.R, 'Predictions')
    error('evaluation.mat must contain R.Predictions.');
end
R = loaded.R;
manifest = readtable(manifestCsv, 'TextType', 'string');
classes = ["crack", "leakage", "abscission", "corrosion", "bulge"];
ids = string(manifest.ImageID);
lookup = containers.Map(cellstr(ids), num2cell(1:numel(ids)));
imageIDs = string(R.ImageIDs);
if numel(imageIDs) ~= numel(R.Predictions.Boxes)
    error('Image IDs and prediction cell counts differ.');
end

records = cell(0, 1);
for i = 1:numel(imageIDs)
    name = char(imageIDs(i));
    if ~isKey(lookup, name)
        error('Image ID not found in manifest: %s', name);
    end
    boxes = double(R.Predictions.Boxes{i});
    labels = string(R.Predictions.Labels{i});
    scores = double(R.Predictions.Scores{i});
    if isempty(boxes)
        continue;
    end
    if size(boxes, 2) ~= 4 || size(boxes, 1) ~= numel(labels) || numel(labels) ~= numel(scores)
        error('Prediction dimensions disagree for image %s.', name);
    end
    for j = 1:numel(scores)
        categoryID = find(classes == lower(strtrim(labels(j))), 1);
        if isempty(categoryID)
            error('Unknown prediction class %s in image %s.', labels(j), name);
        end
        b = boxes(j, :);
        % MATLAB pixel coordinates start at 1. The shared evaluator uses
        % zero-based top-left coordinates in the original image.
        b(1:2) = b(1:2) - 1;
        records{end+1, 1} = struct( ...
            'image_id', lookup(name), ...
            'category_id', categoryID, ...
            'bbox', b, ...
            'score', scores(j)); %#ok<AGROW>
    end
end
if isempty(records)
    output = struct('image_id', {}, 'category_id', {}, 'bbox', {}, 'score', {});
else
    output = vertcat(records{:});
end
fid = fopen(outputJson, 'w');
if fid < 0
    error('Cannot open output file %s.', outputJson);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(output));
fprintf('Exported %d detections for %d images to %s\n', numel(output), numel(imageIDs), outputJson);
end
