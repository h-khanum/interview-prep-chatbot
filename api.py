from flask import request, jsonify
import sqlite3
import json

def init_api(app, chatbot):
    
    @app.route('/api/questions', methods=['GET'])
    def get_questions():
        topic = request.args.get('topic')
        difficulty = request.args.get('difficulty')
        
        conn = sqlite3.connect('interview_prep.db')
        cursor = conn.cursor()
        
        query = 'SELECT * FROM questions WHERE 1=1'
        params = []
        
        if topic:
            query += ' AND topic = ?'
            params.append(topic)
        
        if difficulty:
            query += ' AND difficulty = ?'
            params.append(difficulty)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        questions = []
        for row in rows:
            questions.append({
                'id': row[0],
                'topic': row[1],
                'difficulty': row[2],
                'question': row[3],
                'model_answer': row[4],
                'keywords': row[5].split(',')
            })
        
        return jsonify(questions)
    
    @app.route('/api/questions', methods=['POST'])
    def add_question():
        auth_key = request.headers.get('X-Admin-Key')
        if auth_key != 'admin-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.json
        required_fields = ['topic', 'difficulty', 'question', 'model_answer', 'keywords']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        conn = sqlite3.connect('interview_prep.db')
        cursor = conn.cursor()
        
        keywords_str = ','.join(data['keywords']) if isinstance(data['keywords'], list) else data['keywords']
        
        cursor.execute('''
            INSERT INTO questions (topic, difficulty, question, model_answer, keywords)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['topic'], data['difficulty'], data['question'], 
              data['model_answer'], keywords_str))
        
        conn.commit()
        question_id = cursor.lastrowid
        conn.close()
        
        return jsonify({'status': 'success', 'question_id': question_id}), 201
    
    @app.route('/api/questions/<int:question_id>', methods=['PUT'])
    def update_question(question_id):
        auth_key = request.headers.get('X-Admin-Key')
        if auth_key != 'admin-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.json
        conn = sqlite3.connect('interview_prep.db')
        cursor = conn.cursor()
        
        update_fields = []
        params = []
        
        if 'topic' in data:
            update_fields.append('topic = ?')
            params.append(data['topic'])
        
        if 'difficulty' in data:
            update_fields.append('difficulty = ?')
            params.append(data['difficulty'])
        
        if 'question' in data:
            update_fields.append('question = ?')
            params.append(data['question'])
        
        if 'model_answer' in data:
            update_fields.append('model_answer = ?')
            params.append(data['model_answer'])
        
        if 'keywords' in data:
            keywords_str = ','.join(data['keywords']) if isinstance(data['keywords'], list) else data['keywords']
            update_fields.append('keywords = ?')
            params.append(keywords_str)
        
        if not update_fields:
            return jsonify({'error': 'No fields to update'}), 400
        
        params.append(question_id)
        query = f"UPDATE questions SET {', '.join(update_fields)} WHERE id = ?"
        
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'question_id': question_id})
    
    @app.route('/api/questions/<int:question_id>', methods=['DELETE'])
    def delete_question(question_id):
        auth_key = request.headers.get('X-Admin-Key')
        if auth_key != 'admin-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        
        conn = sqlite3.connect('interview_prep.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM questions WHERE id = ?', (question_id,))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Question deleted'})
    
    @app.route('/api/topics', methods=['GET'])
    def get_topics():
        conn = sqlite3.connect('interview_prep.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT topic FROM questions')
        topics = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return jsonify(topics)
    
    @app.route('/api/init_sample_data', methods=['POST'])
    def init_sample_data():
        auth_key = request.headers.get('X-Admin-Key')
        if auth_key != 'admin-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            with open('data/questions.json', 'r') as f:
                questions = json.load(f)
            
            conn = sqlite3.connect('interview_prep.db')
            cursor = conn.cursor()
            
            for q in questions:
                keywords_str = ','.join(q['keywords'])
                cursor.execute('''
                    INSERT INTO questions (topic, difficulty, question, model_answer, keywords)
                    VALUES (?, ?, ?, ?, ?)
                ''', (q['topic'], q['difficulty'], q['question'], 
                      q['model_answer'], keywords_str))
            
            conn.commit()
            conn.close()
            
            return jsonify({'status': 'success', 'message': f'Loaded {len(questions)} questions'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
