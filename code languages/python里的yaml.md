参考：[python yaml用法详解 - konglingbin - 博客园 (cnblogs.com)](https://www.cnblogs.com/klb561/p/9326677.html)

# 1.YAML是什么？

- 概述
  - YAML，**YAML Ain't Markup Language**，YAML不是**标记语言**
  - 是一种人类可读的**数据序列化**格式
  - 它的设计目标是让**配置文件**和**数据交换**变得简单、易读和易于编写。

## 1.1 标记语言

- 理解
  - 标记语言（**Markup Language**）是一种用于将**文本与标记**结构化的**计算机语言**
  - 简单说就是：**在文本中嵌入特定的标记或标签**
  - 标记语言中的**文本被包围在特定的标记或标签中**，这些**标记描述了文本的结构、样式和语义**
  - 标记语言的主要目的是**指示文本中的元素如何在页面上布局、显示和呈现。**
  - 标记本身通常不会直接显示在最终输出中，而是用于指导呈现引擎或渲染器执行相应的操作。
- 常见的标记语言：
  1. **HTML（HyperText Markup Language）**：用于创建网页的标记语言。HTML使用各种标签来定义文本的结构、链接、图像、表格、表单等元素。浏览器通过解释HTML标记来呈现网页内容。
  2. **XML（eXtensible Markup Language）**：一种通用的标记语言，用于描述数据的结构和内容。XML不仅用于标记文档，还用于数据交换和配置文件。与HTML不同，XML标记没有预定义的语义，可以根据需要定义自己的标签。
  3. **Markdown**：一种轻量级的标记语言，用于将文本转换为HTML或其他格式。Markdown使用简单的标记语法来定义标题、列表、链接、强调等文本样式。
  4. **LaTeX**：一种用于排版科学和技术文档的标记语言。它支持复杂的数学公式、图表、引用等高级排版要求。

## 1.2 数据序列化

- 理解

  - 数据序列化是指将**数据结构或对象**转换为**可以在存储或传输中使用的格式**，以便在稍后**重新创建或重建**原始数据结构。
  - 在计算机科学和编程中，数据序列化是为了**将数据从一种表示形式转换为另一种**，以便能够有效地**存储、传输或共享数据**，同时**保留其结构和语义**。
  - 序列化的过程通常涉及**将数据结构、对象属性和值**转换为**特定格式的字节流或文本字符串**。在需要重新使用数据时，可以对序列化后的数据进行**反序列化**，将其重新转换为原始的数据结构或对象。

- 常见数据序列化格式：

  1. **JSON（JavaScript Object Notation）**：一种轻量级的数据交换格式，易于人类阅读和编写，也易于机器解析和生成。它基于JavaScript对象的表示，但已成为多种编程语言的通用数据格式。

     - 优点：
       - 易于阅读和编写，具有人类可读性。
       - 在多种编程语言中有广泛的支持和解析库。
       - 被广泛用于Web应用程序、API通信等。
     - 缺点：
       - 相对于二进制格式，JSON数据相对冗长，占用更多的存储空间。
       - 对于大型数据集，解析和生成JSON可能会变得较慢。
       - **没有注释**

  2. **XML（eXtensible Markup Language）**：**一种标记语言**，**用于描述数据的结构和内容**。XML被广泛用于数据交换和配置文件，但由于其冗长的语法，相对于JSON而言，更复杂和难以阅读。

     - 优点：
       - 结构化，可以表示复杂的数据层次结构。
       - 支持命名空间，适用于一些特定的行业标准。
       - 被广泛应用于配置文件、文档存储等。

     - 缺点：
       - 语法冗长，对人类来说不够友好。
       - 解析和生成XML相对较慢，需要更多的处理时间。

  3. **YAML（YAML Ain't Markup Language）**：一种人类可读的数据序列化格式，强调易读性和可用性。YAML被广泛用于配置文件和数据交换，特别适用于需要人类编辑和阅读的场景。

     - 优点：
       - 极具可读性，易于阅读和编辑，尤其适合配置文件。
       - 支持注释，可以提供更多的解释和说明。
       - 非常适合人类编辑，也可用于数据交换。

     - 缺点：
       - 由于**其易读性，可能存在一些潜在的安全问题（例如，注入攻击）**。
       - 比二进制格式需要更多的处理时间和解析成本。

  4. **Protocol Buffers（protobuf）**：由Google开发的一种高效的二进制序列化格式，旨在节省存储空间和传输带宽。它定义了数据结构和消息格式，并生成用于多种编程语言的代码。

     - 优点：
       - 二进制格式，紧凑且高效，占用较少的存储空间。
       - 生成的代码可以在多种编程语言中使用，提供了类型安全性。
       - 解析和生成速度快，适用于高性能要求的系统。

     - 缺点：
       - 由于是二进制格式，人类可读性较差。

  5. **MessagePack**：一种高效的二进制序列化格式，类似于JSON，但更紧凑。它可以在多种编程语言之间进行交换。

     - 优点：
       - 二进制格式，紧凑且高效。
       - 在各种编程语言中有支持。
       - 解析和生成速度快。
     - 缺点：
       - 相对于JSON和YAML，在人类可读性方面略有不足。

# 2. YAML格式

- 语法规则

  1. 大小写敏感
  2. 使用**缩进表示层级关系**，使用空格缩进，而非Tab键缩进
  3. 相同层级的元素左侧对齐
  4. **键值对用冒号 “:” **结构表示，冒号与值之间需用空格分隔
  5. **数组前加有 “-”** 符号，符号与值之间需用空格分隔
  6. **None值**可用null 和 ~ 表示
  7. 多组数据之间使用3横杠(---)—分割
  8. #表示注释，但不能在一段代码的行末尾加 #注释，否则会报错

- yaml文件数据格式

  1. 字典

     ```yaml
     #test_字典.yaml
     name: 吴彦祖
     age: 20
     ```

  2. 列表

     ```yaml
     #test_列表.yaml
     - 10
     - 20
     - 30
     ```

  3. 列表和字典互相嵌套

     ```yaml
     #test_列表中的字典.yaml
     -
      name: 吴彦祖
      age: 21
     -
     
     #test_字典中的字典.yaml
     name: 
      name1: 吴彦祖
      
     #test_字典中的列表.yaml
     name:
      - 吴彦祖
      - 周星驰
      - uzi
     age: 20
     ```

  4. 一个yaml文件内有多组数据，用---分隔

     ```yaml
     #test_多组数据.yaml
     - 10
     - 20
     - 30
     ---
     name: 吴彦祖
     age: 20
     ```

# 3 python 里的yaml

- 安装处理库：PyYaml

  `pip install pyyaml`

## 3.1 `yaml.load`和`yaml.dump`

- yaml里核心的**序列化和反序列化**

1. `yaml.load(stream, Loader=None)`

   - **反序列化**：将YAML格式的文本（字符串或文件流）反序列化为**Python对象**

   - 参数：

     1. `stream`: 要解析的YAML格式文本，可以是一个包含**YAML内容的字符串**或**打开的文件对象**。
     2. `Loader`: ~~一个可选的参数~~（**现python3版本是必选参数**），用于**指定YAML解析器的类**。默认情况下，`Loader`设置为`yaml.Loader`，但可以根据需要选择其他加载器，如`yaml.SafeLoader`用于更安全的加载。

   - 示例

     ```python
     import yaml
     
     yaml_text = """
     name: John Doe
     age: 30
     hobbies:
       - Reading
       - Hiking
       - Cooking
     """
     
     data = yaml.load(yaml_text, Loader=yaml.SafeLoader)
     print(data)
     ```

2. `yaml.dump(data, stream=None, Dumper=None, **kwds)`

   - **序列化**：函数用于将**Python对象**序列化为**YAML格式的文本**

   - 参数：

     1. `data`: 要序列化的**Python对象**，通常是一个**字典、列表等数据结构**。
     2. `stream`: **可选参数**，用于指定写入YAML内容的**目标流**，可以是文件对象或类似文件的对象。
     3. `Dumper`: **可选参数**，用于指定YAML生成器的类。默认情况下，`Dumper`设置为`yaml.Dumper`，但可以根据需要选择其他生成器，如`yaml.SafeDumper`用于更安全的转储。
     4. `**kwds`: 其他关键字参数，用于传递给Dumper的初始化

   - 示例：

     ```python
     import yaml
     
     data = {
         'name': 'Jane Smith',
         'age': 25,
         'hobbies': ['Painting', 'Gardening']
     }
     
     yaml_text = yaml.dump(data, default_flow_style=False)
     print(yaml_text)
     ```

- 注意：为了安全起见，通常建议在`load`和`dump`函数中使用`SafeLoader`和`SafeDumper`。这有助于防止一些潜在的安全问题，尤其是在处理不受信任的YAML数据时。
- ==序列化后的YAML格式的文本，本质还是字符串，str==

## 3.2 高阶玩法

- **写这篇文档的动力**

### 3.2.1 yaml.YAMLObject构造器

- 概述
  - `yaml.YAMLObject`是`pyyaml`库中的一个**基类**，允许使用者通过**继承**的方式，创建**自定义类**。
  - 方便的是，通过**继承`yaml.YAMLObject`类**，您可以创建具有**自定义序列化和反序列化行为**的类。
  - 可以**自定义如何将Python对象序列化为YAML格式**以及**如何将YAML格式反序列化为Python对象**。

- 基本用法

  - **创建一个新类**并将其**继承自`yaml.YAMLObject`**。然后，您可以在该类中定义一个属性和两个方法：**`yaml_tag`**、**`to_yaml()`** 和 **`from_yaml()`** 

  1. `yaml_tag`：这是一个类属性，用于**指定在YAML中使用的标签**。**标签**是一个唯一标识符（对象），用于告诉`pyyaml`如何识别和处理该类的对象。
  2. `to_yaml()`：这是一个**实例方法**，用于**将Python对象转换为YAML格式的数据**。您需要在此方法中返回一个包含YAML数据的字符串。
  3. `from_yaml(cls, loader, node)`：这是一个**类方法**，用于**将YAML格式的数据转换为Python对象**。`loader`是一个YAML加载器对象，`node`是从YAML解析得到的节点。

- 示例

  ```python
  import yaml
  from yaml import YAMLObject
  
  class Person(YAMLObject):
      yaml_tag = '!Person'
  
      def __init__(self, name, age):
          self.name = name
          self.age = age
  
      def __repr__(self):
          return f"Person(name={self.name}, age={self.age})"
  
      @classmethod
      def from_yaml(cls, loader, node):
          data = loader.construct_mapping(node)
          return cls(**data)
  
      def to_yaml(self, representer):
          data = {'name': self.name, 'age': self.age}
          return representer.represent_mapping(self.yaml_tag, data)
  
  # Register the class for loading
  yaml.SafeLoader.add_constructor('!Person', Person.from_yaml)
  
  # Creating and using Person objects
  person = Person('John Doe', 30)
  
  yaml_text = yaml.dump(person)
  print(yaml_text)
  
  loaded_person = yaml.load(yaml_text, Loader=yaml.SafeLoader)
  print(loaded_person)
  
  ```

  

