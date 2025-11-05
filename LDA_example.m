% uiopen('owid-covid-data.csv',1)
countries(1)=owidcoviddata.location(1);
c=1;
% d(c)=1;
for i=2:size(owidcoviddata,1)
    if owidcoviddata.location(i) ~= countries(c)
        c=c+1;
        countries(c)=owidcoviddata.location(i);
    %     d(c)=1;
    % else 
    %     d(c)=d(c)+1;
    end
    % deaths(c,d(c))=owidcoviddata.new_deaths_smoothed_per_million(i);
    % dates(c,d(c))=owidcoviddata.date(d(c));
end


for i=1:size(countries,2)
    kk=owidcoviddata(find(owidcoviddata.location==countries(i)),:);
    d(i)=size(kk,1);
    deaths(i,1:size(kk,1))=kk.new_deaths_smoothed;
    dates(i,1:size(kk,1))=kk.date;
end


deaths(isnan(deaths))=0;

[m,c]=max(d);
dates_voc=dates(c,:);

vocabulary=string(dates_voc);


figure
plot(dates_voc,deaths')

mdl=fitlda(floor(deaths),14);

plot(mdl.TopicWordProbabilities)

figure
for i=1:mdl.NumTopics
subplot(mdl.NumTopics,1,i)
plot(dates_voc,mdl.TopicWordProbabilities(:,i))
end
%%% correlation between topics
correlation = corrcoef(mdl.TopicWordProbabilities);

numTopics = mdl.NumTopics;
for i = 1:numTopics
    top = topkwords(mdl,3,i);
    topWords(i) = join(top.Word,", ");
end

figure
heatmap(correlation - eye(numTopics))%, ...
  %  XDisplayLabels=topWords, ...
  %  YDisplayLabels=topWords)

title("LDA Topic Correlations")
xlabel("Topic")
ylabel("Topic")


%%% wordclouds
numTopics = mdl.NumTopics;

figure
tiledlayout("flow")
title("LDA Topics")

for i = 1:numTopics
    nexttile
    wordcloud(mdl,i);
    title("Topic " + i)
end

%%%% visualize mixtures as stacked bar charts

topicMixtures=transform(mdl,floor(deaths));
figure
barh(topicMixtures,'stacked')
xlim([0 1])
title("Topic Mixtures")
xlabel("Topic Probability")
ylabel("Country")
yticks(1:size(countries,2))
yticklabels(countries)
legend("Topic "+ string(1:numTopics),'Location','northeastoutside')


%%% visualizing clusters with tSNE
XY = tsne(mdl.DocumentTopicProbabilities);
[~,topTopics] = max(mdl.DocumentTopicProbabilities,[],2);
% for i = 1:numTopics
%     top = topkwords(mdl,3,i);
%     topWords(i) = join(top.Word,", ");
% end
figure
clr = jet(mdl.NumTopics);
gscatter(XY(:,1),XY(:,2),topTopics,clr,'o*+xds')

title("Topic Mixtures")

legend("Topic "+ string(1:numTopics), ...
    Location="southoutside", ...
    NumColumns=2)


%%% evaluate optimal number of topics
numTopicsRange = 3:40; %[5 10 15 20 40];
for i = 1:numel(numTopicsRange)
    numTopics = numTopicsRange(i)
    
    mdl = fitlda(floor(deaths),numTopics,Verbose=0);
    
    [~,validationPerplexity(i)] = logp(mdl,floor(deaths));
    timeElapsed(i) = mdl.FitInfo.History.TimeSinceStart(end);
end

figure
yyaxis left
plot(numTopicsRange-2,validationPerplexity,"+-")
ylabel("Validation Perplexity")

yyaxis right
plot(numTopicsRange-2,timeElapsed,"o-")
ylabel("Time Elapsed (s)")

legend(["Validation Perplexity" "Time Elapsed (s)"],Location="southeast")
xlabel("Number of Topics")



%%%%%%%%% latent semantic analysis ¿?
mdl1=fitlsa(deaths,5);
figure
for i=1:5
subplot(5,1,i)
plot(dates_voc,mdl1.WordScores(:,i))
end