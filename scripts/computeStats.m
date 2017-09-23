function stats = computeStats(folder_name)
time = [];
cost = [];
expands = [];
stats = []
scenes = 48:59;
% scenes = 48:51
% scenes = [48 49];
% frames = [1 4];
frames = 1:9;
time_axis = 0:0.1:300

scene_costs_mean = []
scene_costs_std = []



for k=scenes
    name = [folder_name '00' num2str(k) '/' ]
    frame_costs = []
    min_times = []
    for i=frames
      stat_filename = [name num2str(i) '_stats.txt'];
      perch_filename = [name num2str(i) '_perch.txt'];
      if exist(stat_filename, 'file')~=2 || exist(perch_filename, 'file')~=2 
        continue;
      end
      stats_data = load(stat_filename);
      perch_data = dataread('file', perch_filename, '%s', 'delimiter', '\n');
      num_objects = length(perch_data) / 6;
      costs = sceneStats(stats_data, num_objects, time_axis);
      frame_costs = [frame_costs; costs];
    end
    avg_costs = mean(frame_costs);
    std_costs = std(frame_costs, 1);
    scene_costs_mean = [scene_costs_mean; avg_costs];
    L = min(avg_costs(~isnan(avg_costs)))
    U = max(avg_costs(~isnan(avg_costs)))
    scene_costs_std = [scene_costs_std; avg_costs];
    min_times = [min_times; stats_data(1,2)];
    fig = figure;
    set(gcf, 'renderer', 'painters')
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 30])
    % plot(time_axis, [frame_costs; avg_costs])
    % plot(time_axis, [avg_costs]); 
    axis equal;
    shadedErrorBar(time_axis, avg_costs, ...
    std_costs, 'lineprops',{'-k','LineWidth', 10}, 'transparent',false);
    % shadedErrorBar(time_axis, avg_costs, ...
    % std_costs, 'lineprops',{'-b','LineWidth', 10});
    hold on;
    % plot([min_time min_time], [min(avg_costs) max(avg_costs)], '--');
    % ax.YLim(2) = ax.YLim(1) + 15.0
    ax = gca;
    ax.XLim = [0 300]
    ax.YLim = [L-8 U+8]
    % set(ax, 'FontSize', 40, 'LineWidth', 5, 'FontName', 'Times');
    set(ax, 'FontSize', 40, 'FontName', 'Times');
    set(gca, 'box', 'off');

    mean_min_time = mean(min_times)
    plot([mean_min_time mean_min_time], [ax.YLim(1) ax.YLim(2)], 'r--',...
    'LineWidth', 5);

    xh = xlabel('Time (s)');
    yh = ylabel('Sol. Cost');
    
    set(xh, 'FontName', 'cmr10', 'interpreter', 'latex');
    set(yh, 'FontName', 'cmr10', 'interpreter', 'latex');
    set(gca, 'FontSize', 50);
    im_name = [num2str(k) '_stats.eps']
    print(gcf, im_name, '-depsc2')
    print(gcf, im_name, '-depsc2')
    % im_name = [num2str(k) '_stats.pdf']
    % print(gcf, im_name, '-dpdf')
    % print(gcf, im_name, '-dpdf')
end
close all

% figure;
% plot(time_axis, scene_costs_mean)
