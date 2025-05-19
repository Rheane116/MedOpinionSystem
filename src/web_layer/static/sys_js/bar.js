function render_bar(data, id){
  var chartDom = document.getElementById(id);
  var myChart = echarts.init(chartDom);
  var option;

  var positiveData = []
  var negativeData = []

   // 遍历 yAxis.data，填充数据
  //var yAxisData = ['诊疗水平', '医学道德', '服务管理', '收费水平', '医疗监管', '社会口碑', '资源/团队', '人员待遇', '环境/设施'];
  var yAxisData = Object.keys(data)
  yAxisData = yAxisData.filter(item => item !== "所有主题");
  yAxisData.forEach(function (category) {
      var sum = data[category][0] + data[category][1];
      var value_p = sum !== 0 ? data[category][0] / sum : 0;
      var value_n = sum !== 0 ? data[category][1] / sum : 0;
      positiveData.push(value_p);
      negativeData.push(value_n);
  });

  option = {
    title: {
      text: ''
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    legend: {},
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'value',
      boundaryGap: [0, 0.01],
      min: 0,  // 明确设置最小值
      max: 1   // 明确设置最大值
    },
    yAxis: {
      type: 'category',
      data: yAxisData
    },
    series: [
      {
        name: '情感正向率',
        type: 'bar',
          data: positiveData,
          itemStyle: {
              color: '#b6e5b1' // 设置柱子颜色为绿色
          }
      },
      {
        name: '情感负向率',
        type: 'bar',
          data: negativeData,
          itemStyle: {
              color: '#ff9e81' // 设置柱子颜色为红色
          }
      }
    ]
  };
  option && myChart.setOption(option);
  myChart.resize(); // 强制重绘        
}

