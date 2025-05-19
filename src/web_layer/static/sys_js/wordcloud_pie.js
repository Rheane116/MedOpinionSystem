var overallWordcloudChart = echarts.init(document.getElementById('wordcloud-overall'));
var overallPieChart = echarts.init(document.getElementById('pie-overall'));
var categoryWordcloudChart = echarts.init(document.getElementById('wordcloud-category'));
var categoryPieChart = echarts.init(document.getElementById('pie-category'));

renderWordcloudCharts(output['所有主体'], overallWordcloudChart, 'wordcloud-overall');
renderPieCharts(output['所有主体'], overallPieChart, 'pie-overall');
renderWordcloudCharts(output['医疗机构'], categoryWordcloudChart, 'wordcloud-category');
renderPieCharts(output['医疗机构'], categoryPieChart, 'pie-category');

// 窗口大小变化时重绘
window.addEventListener('resize', function() {
    overallWordcloudChart && overallWordcloudChart.resize();
    overallPieChart && overallPieChart.resize();
    categoryWordcloudChart && categoryWordcloudChart.resize();
    categoryPieChart && categoryPieChart.resize();
});
    
function renderWordcloudCharts(wordcloudData, chart, id){
    const container = document.getElementById(id);
    console.assert(container, '容器未找到');
    console.assert(container.offsetWidth > 0, '容器宽度为0');
    // 根据 result 字典生成 data 数组
    var data = [];
    for (var key in wordcloudData) {
        if(key == "所有主题"){
            continue;
        }
        const val = wordcloudData[key][2]
        if (wordcloudData.hasOwnProperty(key) && val != 0 && val != -1) {
            data.push({
                name: key,
                value: val
            });
        }
    }

    //console.log('---wordcloud-----')
    //console.log(data)

    // 指定图表的配置项和数据
    var option = {
        //backgroundColor: '#e9ccff',
        title: {
            text: '',
            left: 'center'
        },
        tooltip: {},
        series: [
            {
                type : 'wordCloud',  //类型为字符云
                shape:'circle',  //平滑
                gridSize : 10, //网格尺寸
                size : ['50%','60%'],
                //sizeRange : [ 50, 100 ],
                rotationRange : [-45, 0, 45, 90], //旋转范围
                textStyle : {
                    normal : {
                        fontFamily:'微软雅黑',
                        color: function() {
                            return 'rgb(' +
                                Math.round(Math.random() * 255) +
                        ', ' + Math.round(Math.random() * 255) +
                        ', ' + Math.round(Math.random() * 255) + ')'
                            }
                        },
                    emphasis : {
                        shadowBlur : 5,  //阴影距离
                        shadowColor : '#333'  //阴影颜色
                    }
                },
                left: 'center',
                top: '-10%',
                right: null,
                bottom: null,
                width:'100%',
                height:'100%',
                data: data
            }
        ]
    };
    // 使用刚指定的配置项和数据显示图表。
    chart.setOption(option,true);  
}

function renderPieCharts(pieData, chart, id){

    var chartDom = document.getElementById(id);
    // 原始数据
    var originalData = [];
    for (var key in pieData) {
        if(key == "所有主题"){
            continue;
        }
        if (pieData.hasOwnProperty(key)) {
            originalData.push({
                name: key,
                value: pieData[key][2]
            });
        }
    }
    
    // 当前排序状态 (1: 降序, -1: 升序)
    var sortState = 1;
    
    // 排序函数
    function sortData(data, order) {
        return data.slice().sort(function(a, b) {
            return (b.value - a.value) * order;
        });
    }
    
    // 初始排序数据
    var sortedData = sortData(originalData, sortState);
    
    // 创建排序按钮容器（放在图例上方）
    var sortContainer = document.createElement('div');
    sortContainer.style.position = 'absolute';
    sortContainer.style.left = '5%'; // 与图例左对齐
    sortContainer.style.top = '10%'; // 在图例上方
    sortContainer.style.zIndex = '100';
    sortContainer.style.width = '20%'; // 与图例同宽
    sortContainer.style.display = 'flex';
    sortContainer.style.gap = '5px';
    chartDom.appendChild(sortContainer);
    
    // 创建升序按钮
    var ascBtn = document.createElement('button');
    ascBtn.innerHTML = '↑';
    ascBtn.style.flex = '1';
    ascBtn.style.background = sortState === -1 ? '#e9ecef' : '#f8f9fa';
    ascBtn.style.border = '1px solid ' + (sortState === -1 ? '#adb5bd' : '#dee2e6');
    ascBtn.style.borderRadius = '4px';
    ascBtn.style.padding = '3px 5px';
    ascBtn.style.cursor = 'pointer';
    ascBtn.style.fontSize = '12px';
    ascBtn.style.fontFamily = 'Arial';
    ascBtn.style.color = '#495057';
    ascBtn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';
    ascBtn.style.transition = 'all 0.2s';
    ascBtn.style.textAlign = 'center';
    
    // 创建降序按钮
    var descBtn = document.createElement('button');
    descBtn.innerHTML = '↓';
    descBtn.style.flex = '1';
    descBtn.style.background = sortState === 1 ? '#e9ecef' : '#f8f9fa';
    descBtn.style.border = '1px solid ' + (sortState === 1 ? '#adb5bd' : '#dee2e6');
    descBtn.style.borderRadius = '4px';
    descBtn.style.padding = '3px 5px';
    descBtn.style.cursor = 'pointer';
    descBtn.style.fontSize = '12px';
    descBtn.style.fontFamily = 'Arial';
    descBtn.style.color = '#495057';
    descBtn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';
    descBtn.style.transition = 'all 0.2s';
    descBtn.style.textAlign = 'center';
    
    // 悬停效果
    function setupButtonHover(btn) {
        btn.onmouseover = function() {
            if ((btn === ascBtn && sortState !== -1) || (btn === descBtn && sortState !== 1)) {
                this.style.background = '#e9ecef';
                this.style.borderColor = '#ced4da';
            }
        };
        btn.onmouseout = function() {
            if ((btn === ascBtn && sortState !== -1) || (btn === descBtn && sortState !== 1)) {
                this.style.background = '#f8f9fa';
                this.style.borderColor = '#dee2e6';
            }
        };
    }
    
    setupButtonHover(ascBtn);
    setupButtonHover(descBtn);
    
    sortContainer.appendChild(ascBtn);
    sortContainer.appendChild(descBtn);
    
    // 更新按钮状态
    function updateButtonStates() {
        ascBtn.style.background = sortState === -1 ? '#e9ecef' : '#f8f9fa';
        ascBtn.style.borderColor = sortState === -1 ? '#adb5bd' : '#dee2e6';
        descBtn.style.background = sortState === 1 ? '#e9ecef' : '#f8f9fa';
        descBtn.style.borderColor = sortState === 1 ? '#adb5bd' : '#dee2e6';
    }
    
    // 排序按钮点击事件
    ascBtn.onclick = function() {
        if (sortState !== -1) {
            sortState = -1;
            sortedData = sortData(originalData, sortState);
            updateButtonStates();
            updateChart();
        }
    };
    
    descBtn.onclick = function() {
        if (sortState !== 1) {
            sortState = 1;
            sortedData = sortData(originalData, sortState);
            updateButtonStates();
            updateChart();
        }
    };
    
    function updateChart() {
        var option = {
            title: {
                text: '',
                subtext: '',
                left: 'center'
            },
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b}: {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                left: '5%',
                top: '18%',  // 为排序按钮留出空间
                bottom: '20%',
                itemGap: 6,
                textStyle: {
                    fontSize: 12,
                    fontFamily: 'Arial',
                    fontWeight: 'bold',
                    color: '#555'
                },
                formatter: function(name) {
                    var item = sortedData.find(function(d) {
                        return d.name === name;
                    });
                    return name + '：' + (item ? item.value : '');
                }
            },
            series: [{
                name: '分布数据',
                type: 'pie',
                radius: ['30%', '75%'],
                center: ['60%', '50%'],
                avoidLabelOverlap: true,
                itemStyle: {
                    borderRadius: 10,
                    borderColor: '#fff',
                    borderWidth: 2
                },
                label: {
                    show: false,
                    position: 'center'
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: '18',
                        fontWeight: 'bold'
                    }
                },
                labelLine: {
                    show: false
                },
                data: sortedData
            }],
            grid: {
                top: '10%',
                right: '5%',
                bottom: '10%',
                left: '25%',
                containLabel: true
            }
        };
        
        chart.setOption(option, true);
    }
    
    // 初始渲染
    updateChart();
    
    // 窗口大小变化时重绘
    window.addEventListener('resize', function() {
        chart.resize();
    });        
}



