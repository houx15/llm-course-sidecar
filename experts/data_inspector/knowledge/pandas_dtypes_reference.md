# Pandas数据类型参考

## 描述
本文档提供pandas常见数据类型的说明，帮助理解数据集检查中的类型匹配问题。

## 常见数据类型

### 数值类型
- **int64**: 64位整数，范围 -9,223,372,036,854,775,808 到 9,223,372,036,854,775,807
- **int32**: 32位整数，范围 -2,147,483,648 到 2,147,483,647
- **float64**: 64位浮点数，支持小数和科学计数法
- **float32**: 32位浮点数，精度较低但占用内存更少

### 字符串类型
- **object**: pandas中的默认字符串类型，实际是Python对象
- **string**: pandas 1.0+引入的专用字符串类型，性能更好

### 时间类型
- **datetime64[ns]**: 纳秒精度的日期时间类型
- **timedelta64[ns]**: 时间间隔类型

### 布尔类型
- **bool**: 布尔值，True或False

### 分类类型
- **category**: 用于有限集合的分类数据，节省内存

## 类型转换常见问题

### 问题1: object类型包含数值字符串
**现象**: 列的dtype显示为object，但内容是数值字符串如"123", "45.6"
**原因**: 数据中混入了非数值字符（如"N/A", "null", 空格）
**解决**:
```python
df['column'] = pd.to_numeric(df['column'], errors='coerce')  # 将无法转换的值设为NaN
```

### 问题2: 整数列包含缺失值变成float
**现象**: 原本应该是int64的列变成了float64
**原因**: pandas中整数类型不支持NaN，有缺失值时自动转为float
**解决**: 使用nullable integer类型
```python
df['column'] = df['column'].astype('Int64')  # 注意大写I
```

### 问题3: 日期被识别为字符串
**现象**: 日期列的dtype是object而非datetime64
**原因**: pandas未自动识别日期格式
**解决**:
```python
df['date_column'] = pd.to_datetime(df['date_column'], format='%Y-%m-%d')
```

## 教学场景中的类型要求

### 初级任务（Python基础）
- 通常要求简单类型：int64, float64, object
- 允许一定的类型不匹配，学生需要学习类型转换

### 中级任务（数据清洗）
- 严格要求类型匹配
- 学生需要处理缺失值、异常值
- 可能涉及category类型优化

### 高级任务（数据分析）
- 要求正确的datetime类型用于时间序列分析
- 要求category类型用于分组统计
- 性能优化考虑（int32 vs int64）

## 参考资源
- pandas官方文档: https://pandas.pydata.org/docs/user_guide/basics.html#dtypes
- 数据类型性能对比: https://pandas.pydata.org/docs/user_guide/scale.html
