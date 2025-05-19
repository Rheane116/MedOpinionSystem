function render_table(data, id) {
  const container = document.getElementById(id);
  if (!container) {
      console.error(`找不到ID为"${id}"的容器元素`);
      return;
  }
  
  // 清空容器
  container.innerHTML = '';
  
  // 创建表格元素
  const table = document.createElement('table');
  table.classList.add('table');
  
  // 创建表头
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  
  // 主题列
  const nameHeader = document.createElement('th');
  nameHeader.textContent = '主题';
  headerRow.appendChild(nameHeader);
  
  // 情感正向率列（带排序按钮）
  const valueHeader = document.createElement('th');
  valueHeader.style.position = 'relative'; // 为箭头定位做准备
  
  // 创建排序按钮容器
  const sortContainer = document.createElement('div');
  sortContainer.style.display = 'inline-flex';
  sortContainer.style.alignItems = 'center';
  sortContainer.style.gap = '5px';
  sortContainer.style.marginLeft = '8px';
  
  // 创建向上箭头（升序）
  const upArrow = document.createElement('span');
  upArrow.innerHTML = '↑';
  upArrow.style.cursor = 'pointer';
  upArrow.style.opacity = '0.5';
  upArrow.style.transition = 'opacity 0.2s';
  upArrow.title = '升序排列';
  
  // 创建向下箭头（降序）
  const downArrow = document.createElement('span');
  downArrow.innerHTML = '↓';
  downArrow.style.cursor = 'pointer';
  downArrow.style.opacity = '0.5';
  downArrow.style.transition = 'opacity 0.2s';
  downArrow.title = '降序排列';
  
  // 添加箭头到容器
  sortContainer.appendChild(upArrow);
  sortContainer.appendChild(downArrow);
  
  // 添加文本和排序按钮到表头
  valueHeader.innerHTML = '情感正向率';
  valueHeader.appendChild(sortContainer);
  headerRow.appendChild(valueHeader);
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // 创建表格内容
  const tbody = document.createElement('tbody');
  
  // 转换数据为数组格式便于排序
  const dataArray = [];
  for (const key in data) {
      if(key == "所有主题"){
        continue;
      }
      const sum = data[key][0] + data[key][1];
      if (sum === 0) continue;
      
      const ratio = data[key][0] / sum;
      dataArray.push({
          key: key,
          ratio: ratio,
          positive: data[key][0],
          negative: data[key][1]
      });
  }
  
  // 初始排序状态 (1: 降序, -1: 升序)
  let sortState = 1; // 默认降序
  
  // 排序函数
  function sortData(order) {
      return dataArray.slice().sort((a, b) => {
          return (b.ratio - a.ratio) * order;
      });
  }
  
  // 渲染表格行的函数
  function renderRows(sortedData) {
      tbody.innerHTML = ''; // 清空现有行
      
      sortedData.forEach(item => {
          const row = document.createElement('tr');
          
          // 主题单元格
          const nameCell = document.createElement('td');
          nameCell.textContent = item.key;
          row.appendChild(nameCell);
          
          // 情感正向率单元格
          const valueCell = document.createElement('td');
          valueCell.textContent = item.ratio.toFixed(3);
          row.appendChild(valueCell);
          
          // 根据比例设置行样式
          if (item.ratio < 0.5) {
              row.classList.add('table-danger');
          } else {
              row.classList.add('table-success');
          }
          
          tbody.appendChild(row);
      });
  }
  
  // 初始渲染
  renderRows(sortData(sortState));
  
  // 箭头点击事件
  upArrow.addEventListener('click', () => {
      sortState = -1; // 升序
      renderRows(sortData(sortState));
      upArrow.style.opacity = '1';
      downArrow.style.opacity = '0.5';
  });
  
  downArrow.addEventListener('click', () => {
      sortState = 1; // 降序
      renderRows(sortData(sortState));
      upArrow.style.opacity = '0.5';
      downArrow.style.opacity = '1';
  });
  
  // 初始高亮降序箭头
  downArrow.style.opacity = '1';
  
  // 悬停效果
  upArrow.addEventListener('mouseover', () => {
      if (sortState !== 1) upArrow.style.opacity = '0.8';
  });
  
  upArrow.addEventListener('mouseout', () => {
      if (sortState !== 1) upArrow.style.opacity = '0.5';
  });
  
  downArrow.addEventListener('mouseover', () => {
      if (sortState !== -1) downArrow.style.opacity = '0.8';
  });
  
  downArrow.addEventListener('mouseout', () => {
      if (sortState !== -1) downArrow.style.opacity = '0.5';
  });
  
  table.appendChild(tbody);
  container.appendChild(table);
}